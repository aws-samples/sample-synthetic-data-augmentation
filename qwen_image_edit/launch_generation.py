"""Launch the Qwen Image Edit synthetic-generation step as a SageMaker training job.

``generate_synthetic.py`` does all its own S3 I/O and Rekognition calls, so this is
really "generic GPU batch compute": the job just needs a GPU box, the generation
dependencies (``requirements.txt`` in this dir, auto-installed by the DLC), and a few
environment variables. There are no SageMaker input/output channels — the script
reads source images and writes results straight to S3.

This mirrors ``yolo_training/launch_training.py`` (same PyTorch-estimator pattern).

Note: ``SDA_S3_BUCKET`` is read at import time by ``generate_synthetic.py``, so it is
passed via ``environment=`` (not hyperparameters, which only arrive later as CLI
args).
"""
from pathlib import Path

from sagemaker.pytorch import PyTorch

_THIS_DIR = str(Path(__file__).resolve().parent)

# PyTorch GPU DLC used for the generation job (torch 2.5.1 / py311). torch comes
# from this image; requirements.txt adds diffusers/transformers/etc. on top. This
# is a py311 image (vs. the training job's older 2.1.0/py310 DLC) because the
# generation deps need a newer Python. Override via --instance-type is unrelated;
# bump these if a newer DLC is available in your region.
DEFAULT_FRAMEWORK_VERSION = "2.5.1"
DEFAULT_PY_VERSION = "py311"


def launch_generation_job(
    s3_bucket: str,
    role: str,
    placement: str,
    dataset_prefix: str,
    output_suffix: str,
    scene_augmentation: bool = False,
    max_images: int = 0,
    model_parallel: bool = True,
    instance_type: str = "ml.g5.12xlarge",
    s3_output_path: str | None = None,
    region: str = "us-west-2",
    framework_version: str = DEFAULT_FRAMEWORK_VERSION,
    py_version: str = DEFAULT_PY_VERSION,
    job_name_prefix: str = "qwen-generation",
    wait: bool = True,
):
    """Launch a SageMaker job that runs ``generate_synthetic.py``.

    Args:
        s3_bucket: Bucket the generation script reads/writes (becomes ``SDA_S3_BUCKET``).
        role: SageMaker execution role ARN. Needs S3 rw on the bucket and
            ``rekognition:DetectLabels``.
        placement: "hazardous" or "background" (see generate_synthetic.py).
        dataset_prefix: S3 prefix for the output dataset.
        output_suffix: Suffix for the synthetic subdirs (e.g. "hazardous").
        scene_augmentation: Add time-of-day / ambient variation to the prompt.
        max_images: Cap on images to process (0 = all).
        model_parallel: Shard the model across GPUs. Default True for the 4x A10G
            g5.12xlarge; set False for a single large-memory GPU (H100/B200).
            Warning: the sharded path is slower per image than a single-GPU load
            because activations are copied between devices at each layer boundary
            (inter-device communication). Use it only when the model does not fit
            on one GPU, not for speed.
        instance_type: EC2 instance type.
        s3_output_path: Optional S3 URI for the (unused) model artifact tarball.
        region: AWS region for the job / SDK session.
        framework_version, py_version: PyTorch DLC selection.
        job_name_prefix: Prefix for the SageMaker job name.
        wait: Block until the job finishes and stream logs.
    """
    # Hyperparameters map to CLI args (--key value); values must be strings.
    # SageMaker always emits "--key value" for every hyperparameter (never a bare
    # flag), so the boolean flags are passed as explicit "true"/"false" strings and
    # generate_synthetic.py parses them with a str->bool converter.
    hyperparameters: dict[str, str] = {
        "placement": placement,
        "dataset-prefix": dataset_prefix,
        "output-suffix": output_suffix,
        "max-images": str(max_images),
        "scene-augmentation": str(scene_augmentation).lower(),
        "model-parallel": str(model_parallel).lower(),
    }

    estimator_kwargs = {
        "entry_point": "generate_synthetic.py",
        "source_dir": _THIS_DIR,
        "role": role,
        "instance_type": instance_type,
        "instance_count": 1,
        "framework_version": framework_version,
        "py_version": py_version,
        "hyperparameters": hyperparameters,
        "environment": {
            # Read at import time by generate_synthetic.py -> config.require_bucket().
            "SDA_S3_BUCKET": s3_bucket,
            # HuggingFace model cache; /tmp is writable in the container.
            "HF_HOME": "/tmp/hf",
            "AWS_DEFAULT_REGION": region,
        },
        "base_job_name": job_name_prefix,
        # generate_synthetic.py resolves the sibling modules by bare import; the whole
        # source_dir is uploaded, and requirements.txt in it is auto-installed.
    }
    if s3_output_path:
        estimator_kwargs["output_path"] = s3_output_path

    estimator = PyTorch(**estimator_kwargs)  # type: ignore[arg-type]

    # No input channels: the script reads/writes S3 directly via boto3.
    estimator.fit(wait=wait, logs="All")

    return estimator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch Qwen Image Edit generation as a SageMaker job"
    )
    parser.add_argument("--s3-bucket", required=True,
                        help="Bucket to read/write (sets SDA_S3_BUCKET in the job)")
    parser.add_argument("--role", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--placement", required=True,
                        choices=["hazardous", "background"])
    parser.add_argument("--dataset-prefix", required=True,
                        help="S3 prefix for the output dataset")
    parser.add_argument("--output-suffix", required=True,
                        help="Suffix for synthetic subdirs (e.g. 'hazardous')")
    parser.add_argument("--scene-augmentation", action="store_true",
                        help="Add time-of-day / ambient variation to the prompt")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images to process (0 = all)")
    # model-parallel defaults True (g5.12xlarge); --no-model-parallel for single big GPU.
    mp = parser.add_mutually_exclusive_group()
    mp.add_argument("--model-parallel", dest="model_parallel", action="store_true",
                    help="Shard the model across GPUs (needed when it doesn't fit on "
                         "one). Slower per image than a single-GPU load due to "
                         "inter-device communication; use only when required, not for speed.")
    mp.add_argument("--no-model-parallel", dest="model_parallel", action="store_false",
                    help="Load the whole model on one GPU (H100/B200).")
    parser.set_defaults(model_parallel=True)
    parser.add_argument("--instance-type", default="ml.g5.12xlarge")
    parser.add_argument("--s3-output", default=None,
                        help="S3 URI for the (unused) model artifact tarball")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--no-wait", dest="wait", action="store_false",
                        help="Submit and return without streaming logs")
    parser.set_defaults(wait=True)

    args = parser.parse_args()

    launch_generation_job(
        s3_bucket=args.s3_bucket,
        role=args.role,
        placement=args.placement,
        dataset_prefix=args.dataset_prefix,
        output_suffix=args.output_suffix,
        scene_augmentation=args.scene_augmentation,
        max_images=args.max_images,
        model_parallel=args.model_parallel,
        instance_type=args.instance_type,
        s3_output_path=args.s3_output,
        region=args.region,
        wait=args.wait,
    )
