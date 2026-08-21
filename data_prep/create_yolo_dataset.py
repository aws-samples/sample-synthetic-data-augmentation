#!/usr/bin/env python3
"""
Create YOLO-format dataset from OpenImages data stored in S3.

Converts OpenImages annotations to YOLO format and organizes into:
- images/train_original/, images/val_original/, images/test_original/
- images/train_synthetic/, images/val_synthetic/, images/test_synthetic/ (empty, for later)
- labels/train_original/, labels/val_original/, labels/test_original/
- labels/train_synthetic/, labels/val_synthetic/, labels/test_synthetic/ (empty, for later)
- data.yaml
"""

import logging
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
import polars as pl
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
# Quiet noisy boto loggers
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# Class mapping: OpenImages class name -> YOLO class id
CLASS_MAPPING = {
    "Person": 0,
    "Man": 0,  # Map to person
    "Woman": 0,  # Map to person
    "Train": 1,
}

YOLO_CLASS_NAMES = {
    0: "person",
    1: "train",
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Create YOLO dataset from OpenImages S3 data"
    )
    parser.add_argument(
        # Env lookup mirrors qwen_image_edit/config.py (kept inline: this script
        # runs outside that package).
        "--s3-bucket",
        type=str,
        default=os.environ.get("SDA_S3_BUCKET"),
        help="S3 bucket name (defaults to $SDA_S3_BUCKET)",
    )
    parser.add_argument(
        "--source-prefix",
        type=str,
        default="datasets/openimages/train/images/train/",
        help="S3 prefix for source images",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="datasets/openimages_subset/",
        help="S3 prefix for output dataset",
    )
    parser.add_argument(
        "--annotations-prefix",
        type=str,
        default="datasets/openimages/annotations/",
        help="S3 prefix for annotation files",
    )
    parser.add_argument(
        "--aws-profile", type=str, default=None, help="AWS profile name"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print actions without executing"
    )
    return parser.parse_args()


def load_annotations(
    bucket: str, annotations_prefix: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load and cache OpenImages annotations for train/val/test splits."""
    cache_dir = os.path.expanduser("~/.cache/openimages")
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "oidv6-train-annotations-bbox.csv")
    val_cache = os.path.join(cache_dir, "validation-annotations-bbox.csv")
    test_cache = os.path.join(cache_dir, "test-annotations-bbox.csv")
    class_cache = os.path.join(cache_dir, "class-descriptions-boxable.csv")

    # Train annotations
    if not os.path.exists(train_cache):
        print("Downloading train annotations from S3...")
        s3_path = f"s3://{bucket}/{annotations_prefix}oidv6-train-annotations-bbox.csv"
        train_annotations = pl.read_csv(s3_path)
        train_annotations.write_csv(train_cache)
    else:
        print(f"Loading cached train annotations from {train_cache}")
        train_annotations = pl.read_csv(train_cache)

    # Validation annotations
    if not os.path.exists(val_cache):
        print("Downloading validation annotations from S3...")
        s3_path = f"s3://{bucket}/{annotations_prefix}validation-annotations-bbox.csv"
        val_annotations = pl.read_csv(s3_path)
        val_annotations.write_csv(val_cache)
    else:
        print(f"Loading cached validation annotations from {val_cache}")
        val_annotations = pl.read_csv(val_cache)

    # Test annotations
    if not os.path.exists(test_cache):
        print("Downloading test annotations from S3...")
        s3_path = f"s3://{bucket}/{annotations_prefix}test-annotations-bbox.csv"
        test_annotations = pl.read_csv(s3_path)
        test_annotations.write_csv(test_cache)
    else:
        print(f"Loading cached test annotations from {test_cache}")
        test_annotations = pl.read_csv(test_cache)

    # Class descriptions
    if not os.path.exists(class_cache):
        print("Downloading class descriptions from S3...")
        s3_path = f"s3://{bucket}/{annotations_prefix}class-descriptions-boxable.csv"
        class_descriptions = pl.read_csv(s3_path, has_header=False)
        class_descriptions.write_csv(class_cache)
    else:
        print(f"Loading cached class descriptions from {class_cache}")
        class_descriptions = pl.read_csv(class_cache)

    return train_annotations, val_annotations, test_annotations, class_descriptions


def get_class_id(class_descriptions: pl.DataFrame, class_name: str) -> str:
    """Get OpenImages class ID for a class name."""
    return class_descriptions.filter(pl.col("column_2") == class_name)["column_1"][0]


def build_openimages_to_yolo(class_descriptions: pl.DataFrame) -> dict[str, int]:
    """Map OpenImages class codes -> YOLO class ids once (CLASS_MAPPING is static).

    Computed a single time and reused across all images/threads, rather than
    re-filtering the class-descriptions frame for every image.
    """
    return {
        get_class_id(class_descriptions, class_name): yolo_id
        for class_name, yolo_id in CLASS_MAPPING.items()
    }


def get_downloaded_image_ids(s3_client, bucket: str, prefix: str) -> set[str]:
    """Get set of image IDs that exist in S3."""
    downloaded_ids = set()
    paginator = s3_client.get_paginator("list_objects_v2")

    print(f"Listing images in s3://{bucket}/{prefix}...")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            image_id = obj["Key"].split("/")[-1].replace(".jpg", "")
            downloaded_ids.add(image_id)

    print(f"Found {len(downloaded_ids)} images")
    return downloaded_ids


def get_train_class_image_ids(
    annotations: pl.DataFrame, class_descriptions: pl.DataFrame
) -> set[str]:
    """Get image IDs that contain Train class from given annotations."""
    train_class_id = get_class_id(class_descriptions, "Train")
    return set(
        annotations.filter(pl.col("LabelName") == train_class_id)["ImageID"].to_list()
    )


def convert_to_yolo_format(
    xmin: float, xmax: float, ymin: float, ymax: float
) -> tuple[float, float, float, float]:
    """
    Convert OpenImages bbox format to YOLO format.

    OpenImages: XMin, XMax, YMin, YMax (normalized 0-1)
    YOLO: x_center, y_center, width, height (normalized 0-1)
    """
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    return x_center, y_center, width, height


def create_yolo_label(
    annotations: pl.DataFrame, image_id: str, openimages_to_yolo: dict[str, int]
) -> str:
    """Create YOLO label file content for an image.

    ``openimages_to_yolo`` is the prebuilt OpenImages-code -> YOLO-id map (see
    ``build_openimages_to_yolo``), passed in so it is not recomputed per image.
    """
    # Get annotations for this image
    image_annotations = annotations.filter(pl.col("ImageID") == image_id)

    lines = []
    for row in image_annotations.iter_rows(named=True):
        label_name = row["LabelName"]
        if label_name not in openimages_to_yolo:
            continue

        yolo_class_id = openimages_to_yolo[label_name]
        x_center, y_center, width, height = convert_to_yolo_format(
            row["XMin"], row["XMax"], row["YMin"], row["YMax"]
        )
        lines.append(
            f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return "\n".join(lines)


def create_data_yaml(include_synthetic: bool = False) -> str:
    """Create data.yaml content for YOLO training."""
    if include_synthetic:
        yaml_content = """# OpenImages subset - Train detection dataset (with synthetic augmentation)
# For SageMaker training, path will be set to /opt/ml/input/data/training or similar

path: .
train:
  - images/train_original
  - images/train_synthetic
val:
  - images/val_original
  - images/val_synthetic
test:
  - images/test_original
  - images/test_synthetic

names:
"""
    else:
        yaml_content = """# OpenImages subset - Train detection dataset (original only)
# For SageMaker training, path will be set to /opt/ml/input/data/training or similar

path: .
train: images/train_original
val: images/val_original
test: images/test_original

names:
"""
    for class_id, class_name in sorted(YOLO_CLASS_NAMES.items()):
        yaml_content += f"  {class_id}: {class_name}\n"

    return yaml_content


def process_single_image(
    s3_client,
    bucket: str,
    source_prefix: str,
    output_prefix: str,
    split_name: str,
    image_id: str,
    annotations: pl.DataFrame,
    openimages_to_yolo: dict[str, int],
):
    """Process a single image: copy to destination and create label."""
    source_key = f"{source_prefix}{image_id}.jpg"
    dest_image_key = f"{output_prefix}images/{split_name}/{image_id}.jpg"
    label_content = create_yolo_label(annotations, image_id, openimages_to_yolo)
    dest_label_key = f"{output_prefix}labels/{split_name}/{image_id}.txt"

    s3_client.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": source_key},
        Key=dest_image_key,
    )
    s3_client.put_object(
        Bucket=bucket, Key=dest_label_key, Body=label_content.encode("utf-8")
    )
    return image_id


def upload_dataset(
    s3_client,
    bucket: str,
    source_prefixes: dict[str, str],
    output_prefix: str,
    annotations_by_split: dict[str, pl.DataFrame],
    class_descriptions: pl.DataFrame,
    image_ids_by_split: dict[str, list[str]],
    dry_run: bool = False,
    max_workers: int = 16,
):
    """Upload images and labels to S3 in YOLO format."""
    splits = [
        ("train_original", "train"),
        ("val_original", "validation"),
        ("test_original", "test"),
    ]

    # Create empty synthetic directories (placeholder files)
    synthetic_splits = ["train_synthetic", "val_synthetic", "test_synthetic"]

    # Static OpenImages-code -> YOLO-id map, built once and reused across all images.
    openimages_to_yolo = build_openimages_to_yolo(class_descriptions)

    for split_name, split_key in splits:
        image_ids = image_ids_by_split[split_key]
        annotations = annotations_by_split[split_key]
        source_prefix = source_prefixes[split_key]
        print(f"\nProcessing {split_name}: {len(image_ids)} images", flush=True)

        if dry_run:
            for i, image_id in enumerate(image_ids[:3]):
                source_key = f"{source_prefix}{image_id}.jpg"
                dest_image_key = f"{output_prefix}images/{split_name}/{image_id}.jpg"
                label_content = create_yolo_label(
                    annotations, image_id, openimages_to_yolo
                )
                dest_label_key = f"{output_prefix}labels/{split_name}/{image_id}.txt"
                print(
                    f"  Would copy: s3://{bucket}/{source_key} -> s3://{bucket}/{dest_image_key}"
                )
                print(f"  Would create label: s3://{bucket}/{dest_label_key}")
                print(f"  Label content:\n{label_content[:200]}...")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_single_image,
                        s3_client,
                        bucket,
                        source_prefix,
                        output_prefix,
                        split_name,
                        image_id,
                        annotations,
                        openimages_to_yolo,
                    )
                    for image_id in image_ids
                ]
                failed = 0
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc=f"  {split_name}"
                ):
                    try:
                        future.result()
                    except Exception as e:
                        # Log and skip a single bad key rather than aborting the split.
                        failed += 1
                        logger.warning(f"Failed to process an image in {split_name}: {e}")
                if failed:
                    print(f"  {split_name}: {failed} image(s) failed and were skipped")

    # Create placeholder for synthetic directories
    for split_name in synthetic_splits:
        placeholder_key = f"{output_prefix}images/{split_name}/.gitkeep"
        label_placeholder_key = f"{output_prefix}labels/{split_name}/.gitkeep"

        if dry_run:
            print(f"Would create placeholder: s3://{bucket}/{placeholder_key}")
            print(f"Would create placeholder: s3://{bucket}/{label_placeholder_key}")
        else:
            s3_client.put_object(Bucket=bucket, Key=placeholder_key, Body=b"")
            s3_client.put_object(Bucket=bucket, Key=label_placeholder_key, Body=b"")

    # Upload data.yaml (original only)
    data_yaml = create_data_yaml(include_synthetic=False)
    data_yaml_key = f"{output_prefix}data.yaml"

    # Upload data-with-synthetic.yaml
    data_yaml_synthetic = create_data_yaml(include_synthetic=True)
    data_yaml_synthetic_key = f"{output_prefix}data-with-synthetic.yaml"

    if dry_run:
        print(f"\nWould create data.yaml at s3://{bucket}/{data_yaml_key}")
        print(f"Content:\n{data_yaml}")
        print(
            f"\nWould create data-with-synthetic.yaml at s3://{bucket}/{data_yaml_synthetic_key}"
        )
        print(f"Content:\n{data_yaml_synthetic}")
    else:
        s3_client.put_object(
            Bucket=bucket, Key=data_yaml_key, Body=data_yaml.encode("utf-8")
        )
        print(f"\nUploaded data.yaml to s3://{bucket}/{data_yaml_key}")
        s3_client.put_object(
            Bucket=bucket,
            Key=data_yaml_synthetic_key,
            Body=data_yaml_synthetic.encode("utf-8"),
        )
        print(
            f"Uploaded data-with-synthetic.yaml to s3://{bucket}/{data_yaml_synthetic_key}"
        )


def upload_openimages_annotations(
    s3_client,
    bucket: str,
    annotations_prefix: str,
    output_prefix: str,
    annotations_by_split: dict[str, pl.DataFrame],
    class_descriptions: pl.DataFrame,
    image_ids_by_split: dict[str, list[str]],
    dry_run: bool = False,
):
    """Upload filtered OpenImages annotations for each split."""
    # Keep the same filenames load_annotations() expects, so a re-run pointed at the
    # output prefix finds them (train uses the oidv6- prefix, matching the downloader).
    split_to_filename = {
        "train": "oidv6-train-annotations-bbox.csv",
        "validation": "validation-annotations-bbox.csv",
        "test": "test-annotations-bbox.csv",
    }

    print("\nUploading filtered OpenImages annotations...")

    for split_key, annotations in annotations_by_split.items():
        image_ids = set(image_ids_by_split[split_key])
        # Filter annotations to only include images in our subset
        filtered = annotations.filter(pl.col("ImageID").is_in(list(image_ids)))
        filename = split_to_filename[split_key]
        dest_key = f"{output_prefix}annotations/{filename}"

        if dry_run:
            print(
                f"  Would upload {len(filtered)} annotations to s3://{bucket}/{dest_key}"
            )
        else:
            csv_content = filtered.write_csv()
            s3_client.put_object(
                Bucket=bucket, Key=dest_key, Body=csv_content.encode("utf-8")
            )
            print(
                f"  Uploaded {filename} ({len(filtered)} rows) to s3://{bucket}/{dest_key}"
            )

    # Upload class descriptions
    class_desc_key = f"{output_prefix}annotations/class-descriptions-boxable.csv"
    if dry_run:
        print(f"  Would upload class descriptions to s3://{bucket}/{class_desc_key}")
    else:
        csv_content = class_descriptions.write_csv()
        s3_client.put_object(
            Bucket=bucket, Key=class_desc_key, Body=csv_content.encode("utf-8")
        )
        print(
            f"  Uploaded class-descriptions-boxable.csv to s3://{bucket}/{class_desc_key}"
        )


def main():
    args = get_args()

    if not args.s3_bucket:
        raise SystemExit(
            "No S3 bucket configured. Set SDA_S3_BUCKET or pass --s3-bucket."
        )

    # Setup S3 client with increased connection pool for parallel uploads
    if args.aws_profile:
        session = boto3.Session(profile_name=args.aws_profile)
    else:
        session = boto3.Session()
    config = Config(max_pool_connections=50)
    s3_client = session.client("s3", config=config)

    # Load annotations for all splits
    train_ann, val_ann, test_ann, class_descriptions = load_annotations(
        args.s3_bucket, args.annotations_prefix
    )
    annotations_by_split = {
        "train": train_ann,
        "validation": val_ann,
        "test": test_ann,
    }

    # Source prefixes for each split
    base_prefix = args.source_prefix.rstrip("/")
    # Path structure from the downloader: .../<class>/images/{split}/
    # (e.g. datasets/openimages/train/images/train/ for the Train class).
    # --source-prefix points at the train split; swap the final component to reach
    # the validation/test splits.
    source_prefixes = {
        "train": f"{base_prefix}/",
        "validation": base_prefix.rsplit("/", 1)[0] + "/validation/",
        "test": base_prefix.rsplit("/", 1)[0] + "/test/",
    }

    # Get valid image IDs for each split (downloaded AND contain Train class)
    image_ids_by_split = {}
    for split_key, annotations in annotations_by_split.items():
        source_prefix = source_prefixes[split_key]
        downloaded_ids = get_downloaded_image_ids(
            s3_client, args.s3_bucket, source_prefix
        )
        train_class_ids = get_train_class_image_ids(annotations, class_descriptions)
        valid_ids = list(downloaded_ids & train_class_ids)
        image_ids_by_split[split_key] = valid_ids
        print(
            f"{split_key}: {len(valid_ids)} valid images (downloaded & contain Train)"
        )

    # Upload dataset
    upload_dataset(
        s3_client,
        args.s3_bucket,
        source_prefixes,
        args.output_prefix,
        annotations_by_split,
        class_descriptions,
        image_ids_by_split,
        dry_run=args.dry_run,
    )

    # Upload filtered OpenImages annotations
    upload_openimages_annotations(
        s3_client,
        args.s3_bucket,
        args.annotations_prefix,
        args.output_prefix,
        annotations_by_split,
        class_descriptions,
        image_ids_by_split,
        dry_run=args.dry_run,
    )

    print("\nDone!")
    if args.dry_run:
        print("(This was a dry run - no files were actually uploaded)")


if __name__ == "__main__":
    main()
