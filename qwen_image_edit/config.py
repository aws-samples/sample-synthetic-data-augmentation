"""Shared configuration for the synthetic data augmentation pipeline.

The S3 bucket is read from the ``SDA_S3_BUCKET`` environment variable so the
same code runs against any bucket without edits:

    export SDA_S3_BUCKET=amzn-s3-demo-bucket

Every CLI script exposes the same ``--s3-bucket`` flag, which defaults to this
variable: ``data_prep/create_yolo_dataset.py``, ``data_prep/download_openimages.py``,
``evaluation/download_testdata.py``, and ``qwen_image_edit/launch_generation.py``.
Those living outside the ``qwen_image_edit`` package reimplement the trivial env
lookup inline (with a cross-reference comment) rather than importing this module,
since they are run as standalone scripts and can't import it. ``generate_synthetic.py``
requires this variable via ``require_bucket()`` below with no CLI override (it is
the batch entry point; ``launch_generation.py`` sets the variable for the job).
"""
import os

ENV_VAR = "SDA_S3_BUCKET"


def default_bucket() -> str | None:
    """Return the configured bucket, or ``None`` if the env var is unset.

    Suitable as an argparse/click default so ``--s3-bucket`` can still override it.
    """
    return os.environ.get(ENV_VAR)


def require_bucket() -> str:
    """Return the configured bucket or raise a clear error if it is unset.

    Use this where there is no CLI override (e.g. notebooks, batch scripts).
    """
    bucket = os.environ.get(ENV_VAR)
    if not bucket:
        raise RuntimeError(
            f"{ENV_VAR} is not set. Point it at your S3 bucket, e.g.:\n"
            f"    export {ENV_VAR}=amzn-s3-demo-bucket"
        )
    return bucket
