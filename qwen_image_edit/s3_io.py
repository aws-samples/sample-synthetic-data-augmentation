"""S3 image I/O, resizing, and resume-cache helpers."""
import json
import os
from io import BytesIO

from PIL import Image


class ImageResizer:
    """Resize an image so its shorter side is ``min_dim``, and restore it later.

    Qwen Image Edit runs at a controlled resolution; we downscale on the way in
    and upscale the result back to the original dimensions.
    """

    def __init__(self, min_dim: int = 512):
        self.min_dim = min_dim
        self.original_size: tuple[int, int] | None = None

    def resize(self, image: Image.Image) -> Image.Image:
        self.original_size = image.size
        w, h = image.size
        scale = self.min_dim / min(w, h)
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size, Image.LANCZOS)

    def restore(self, image: Image.Image) -> Image.Image:
        if self.original_size is None:
            raise ValueError("No original size stored. Call resize() first.")
        return image.resize(self.original_size, Image.LANCZOS)


def read_image_from_s3(s3_client, bucket: str, image_id: str, prefix: str) -> Image.Image:
    """Load ``<prefix><image_id>.jpg`` from S3 as a PIL image."""
    key = f"{prefix}{image_id}.jpg"
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return Image.open(BytesIO(response["Body"].read()))


def load_cache(cache_file: str) -> dict:
    """Load the resume cache, or return an empty one if it is missing/corrupt.

    Always returns a dict with both ``processed_ids`` and ``failed_ids`` keys, so a
    truncated or older cache can't KeyError the caller.
    """
    cache = {"processed_ids": [], "failed_ids": []}
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                loaded = json.load(f)
            cache["processed_ids"] = list(loaded.get("processed_ids", []))
            cache["failed_ids"] = list(loaded.get("failed_ids", []))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not read cache {cache_file} ({e}); starting fresh")
    return cache


def save_cache(cache: dict, cache_file: str) -> None:
    """Persist the resume cache to disk atomically (write temp, then rename).

    A crash mid-write can't corrupt the existing cache: the rename is atomic on the
    same filesystem, so readers see either the old or the new file, never a partial.
    """
    tmp_file = f"{cache_file}.tmp"
    with open(tmp_file, "w") as f:
        json.dump(cache, f)
    os.replace(tmp_file, cache_file)
