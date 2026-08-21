"""Qwen Image Edit model loading and inference.

On a single large-memory GPU (e.g. an 80GB H100 or a B200) the whole
Qwen-Image-Edit-2509 pipeline fits in bf16, so it loads onto one device with no
manual layer sharding.
Multi-GPU instances with less memory per device (e.g. g5.12xlarge, 4x A10G) cannot
fit the model on one device, so ``load_pipeline(model_parallel=True)`` hand-shards
the transformer's layers across the visible GPUs. Both paths are hidden behind the
same ``load_pipeline`` / ``edit_image`` interface.
"""
import importlib.metadata
import importlib.util
from contextlib import contextmanager

import torch

MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
DEFAULT_INFERENCE_STEPS = 40
DEFAULT_CFG_SCALE = 4.0
DEFAULT_MIN_DIM = 512

# flash-attn 2.7.0 added the symbols (``_wrapped_flash_attn_backward``/``_forward``)
# that diffusers 0.36+ imports in ``attention_dispatch``. Older builds pass diffusers'
# ``>=2.6.3`` guard but crash the import.
_MIN_FLASH_ATTN = (2, 7, 0)

# Layers per GPU when sharding across a 4x A10G g5.12xlarge (60 transformer blocks
# total). GPU 0 is kept light to leave room for the text encoder; the last GPU holds
# the output projection (norm_out/proj_out) and takes no transformer blocks. The VAE
# and remaining components are placed by the pipeline's device_map="balanced".
DEFAULT_LAYER_DISTRIBUTION = (5, 25, 30, 0)


def _build_transformer_device_map(layer_distribution):
    """Map transformer submodules to GPUs for manual model-parallel sharding.

    ``diffusers``' ``device_map="balanced"`` is not robust for this model (it can
    spill layers onto CPU), so we place blocks explicitly across the GPUs listed
    in ``layer_distribution`` (one entry per GPU giving its block count).
    """
    device_map = {
        "pos_embed": 0,
        "time_text_embed": 0,
        "txt_norm": 0,
        "img_in": 0,
        "txt_in": 0,
    }

    layer_idx = 0
    for gpu, count in enumerate(layer_distribution):
        for _ in range(count):
            device_map[f"transformer_blocks.{layer_idx}"] = gpu
            layer_idx += 1

    last_gpu = len(layer_distribution) - 1
    device_map["norm_out"] = last_gpu
    device_map["proj_out"] = last_gpu
    return device_map


@contextmanager
def _hide_incompatible_flash_attn():
    """Make an incompatible flash-attn invisible while diffusers initializes.

    The SageMaker PyTorch DLC ships flash-attn 2.6.3, which passes diffusers'
    ``>=2.6.3`` check but lacks symbols added in 2.7.0. Hiding only that package
    from diffusers' optional-dependency discovery makes it select standard
    attention without uninstalling or otherwise modifying the environment.
    """
    if importlib.util.find_spec("flash_attn") is None:
        yield
        return

    try:
        version = importlib.metadata.version("flash_attn")
        # Strip any local/build suffix (e.g. "2.7.0+cu122", "2.7.0.dev0") before
        # parsing so a decorated-but-compatible version remains available.
        numeric = version.split("+")[0]
        parts = []
        for part in numeric.split(".")[:3]:
            if not part.isdigit():
                break
            parts.append(int(part))
        if tuple(parts) >= _MIN_FLASH_ATTN:
            yield
            return
    except Exception:
        version = "unknown"  # unparseable version: safely disable the integration

    print(
        f"Ignoring incompatible flash-attn {version} while diffusers initializes "
        "(needs >= 2.7.0; using standard attention)..."
    )
    original_find_spec = importlib.util.find_spec

    def find_spec_without_flash_attn(name, package=None):
        if name == "flash_attn":
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = find_spec_without_flash_attn
    try:
        yield
    finally:
        importlib.util.find_spec = original_find_spec


def load_pipeline(
    device: str = "cuda",
    *,
    model_parallel: bool = False,
    layer_distribution=DEFAULT_LAYER_DISTRIBUTION,
):
    """Load the Qwen Image Edit pipeline in bf16.

    :param device: target device when loading onto a single GPU (ignored when
        ``model_parallel`` is set).
    :param model_parallel: when ``True``, hand-shard the transformer across the
        visible GPUs for instances that can't fit the model on one device
        (e.g. 4x A10G g5.12xlarge). When ``False`` (default), load the whole
        pipeline onto ``device`` — suitable for an 80GB H100/B200.
    :param layer_distribution: per-GPU transformer block counts used only when
        ``model_parallel`` is set; must sum to the model's block count.
    """
    with _hide_incompatible_flash_attn():
        from diffusers import QwenImageEditPlusPipeline

    if model_parallel:
        from diffusers.models.transformers import QwenImageTransformer2DModel

        device_map = _build_transformer_device_map(layer_distribution)
        print(
            f"Loading {MODEL_ID} sharded across {len(layer_distribution)} GPUs "
            f"(layers per GPU: {tuple(layer_distribution)})..."
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
            device_map="balanced",
        )
    else:
        print(f"Loading {MODEL_ID} onto {device}...")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        pipeline.to(device)

    pipeline.set_progress_bar_config(disable=None)
    print("Pipeline loaded.")
    return pipeline


def edit_image(
    pipeline,
    image,
    prompt: str,
    *,
    device: str = "cuda",
    seed: int | None = None,
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
    cfg_scale: float = DEFAULT_CFG_SCALE,
):
    """Run a single edit and return the generated PIL image.

    :param seed: per-image seed for reproducibility; ``None`` leaves it random.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": generator,
        "true_cfg_scale": cfg_scale,
        # A single space (not "") is the conventional empty negative prompt for
        # Qwen-Image-Edit: it still tokenizes so true-CFG has an unconditional
        # branch to steer against, without adding negative concepts.
        "negative_prompt": " ",
        "num_inference_steps": num_inference_steps,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
    return output.images[0]
