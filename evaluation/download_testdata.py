"""Download test data from S3 and create YOLO validation dataset YAML."""
import os
from pathlib import Path

import boto3
import click
import yaml

# Default S3 paths. The bucket defaults to $SDA_S3_BUCKET; this env lookup mirrors
# qwen_image_edit/config.py (kept inline: this script runs outside that package).
DEFAULT_BUCKET = os.environ.get("SDA_S3_BUCKET")
DEFAULT_IMAGES_PREFIX = "datasets/openimages_subset/images/test_original/"
DEFAULT_LABELS_PREFIX = "datasets/openimages_subset/labels/test_original/"


def download_s3_folder(bucket: str, prefix: str, local_dir: Path) -> None:
    """Download all objects from an S3 prefix to a local directory."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    local_dir.mkdir(parents=True, exist_ok=True)

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            relative_path = key[len(prefix) :].lstrip("/")
            if not relative_path:
                continue

            target_file = local_dir / relative_path
            target_file.parent.mkdir(parents=True, exist_ok=True)

            print(f"Downloading: {key} -> {target_file}")
            s3.download_file(bucket, key, str(target_file))

    print(f"Downloaded {prefix} to {local_dir}")


def create_val_yaml(
    output_path: str,
    images_subdir: str,
    class_names: list[str],
) -> str:
    """Create a YOLO dataset YAML for validation mode."""
    config = {
        "path": ".",
        "train": images_subdir,
        "val": images_subdir,
        "test": images_subdir,
        "names": dict(enumerate(class_names)),
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created validation YAML: {output_path}")
    return str(output_file)


@click.command()
@click.option("--s3-bucket", "bucket", default=DEFAULT_BUCKET, help="S3 bucket name (defaults to $SDA_S3_BUCKET)")
@click.option("--images-prefix", default=DEFAULT_IMAGES_PREFIX, help="S3 prefix for images")
@click.option("--labels-prefix", default=DEFAULT_LABELS_PREFIX, help="S3 prefix for labels")
@click.option("--local-dir", default="./test_data", help="Local directory to download to")
@click.option("--yaml-output", default="val_dataset.yaml", help="Output path for YAML file")
@click.option("--classes", default="person,train", help="Comma-separated class names")
def main(
    bucket: str,
    images_prefix: str,
    labels_prefix: str,
    local_dir: str,
    yaml_output: str,
    classes: str,
) -> None:
    """Download test data (images + labels) from S3 and create YOLO validation YAML."""
    if not bucket:
        raise click.UsageError(
            "No S3 bucket configured. Set SDA_S3_BUCKET or pass --s3-bucket."
        )

    local_path = Path(local_dir)

    # Download images
    images_local = local_path / "images" / "test_original"
    print(f"Downloading images from s3://{bucket}/{images_prefix}")
    download_s3_folder(bucket, images_prefix, images_local)

    # Download labels
    labels_local = local_path / "labels" / "test_original"
    print(f"Downloading labels from s3://{bucket}/{labels_prefix}")
    download_s3_folder(bucket, labels_prefix, labels_local)

    # Parse class names and create YAML
    class_names = [c.strip() for c in classes.split(",")]
    create_val_yaml(
        output_path=str(local_path / yaml_output),
        images_subdir="images/test_original",
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
