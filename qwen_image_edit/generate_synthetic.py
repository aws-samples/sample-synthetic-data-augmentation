#!/usr/bin/env python3
"""
Batch synthetic image generation using Qwen Image Edit.

Converts the notebook pipeline into a standalone script for unattended runs
on SageMaker notebook instances (via nohup or screen).

Supports 4 ablation conditions via --placement and --scene-augmentation:
  1. hazardous + no-scene   (original "synthetic" condition)
  2. hazardous + scene
  3. background + no-scene
  4. background + scene     (original "supplement" condition)

Requires SDA_S3_BUCKET to point at your bucket (see qwen_image_edit/config.py).
From the repository root, install dependencies with
`uv sync --frozen --project scripts` before running.

Usage from the repository root:
    export SDA_S3_BUCKET=amzn-s3-demo-bucket

    # Run (e.g. background + scene augmentation):
    nohup scripts/.venv/bin/python qwen_image_edit/generate_synthetic.py \
        --placement background \
        --scene-augmentation \
        --dataset-prefix datasets/ablation_background_scene \
        --output-suffix background_scene \
        > generation_bg_scene.log 2>&1 &
"""

import argparse
import os
from io import BytesIO

import boto3
import numpy as np
import polars as pl
from tqdm import tqdm

from config import require_bucket
from model import DEFAULT_MIN_DIM, edit_image, load_pipeline
from prompts import build_prompt
from recognition import dedupe_person_boxes, extract_person_boxes
from s3_io import ImageResizer, load_cache, read_image_from_s3, save_cache

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S3_BUCKET = require_bucket()
SOURCE_DATASET = "datasets/openimages_subset/"
SOURCE_IMAGES = SOURCE_DATASET + "images/train_original/"
SOURCE_LABELS = SOURCE_DATASET + "labels/train_original/"

YOLO_CLASS_NAMES = {0: "person", 1: "train"}

# Where to cache annotation CSVs and the resume state (override via CACHE_DIR).
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.expanduser("~/.cache/synthetic_augmentation"))

# Diffusion seed base; each image adds its index so samples differ but the run
# is reproducible.
SEED_BASE = 2025


# ---------------------------------------------------------------------------
# Dataset loading (from notebook)
# ---------------------------------------------------------------------------

def load_annotations_cached(s3_client):
    cache_dir = CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    ann_cache = os.path.join(cache_dir, "oidv6-train-annotations-bbox.csv")
    cls_cache = os.path.join(cache_dir, "class-descriptions-boxable.csv")

    if not os.path.exists(ann_cache):
        print("Downloading train annotations from S3...")
        train_annotations = pl.read_csv(
            f"s3://{S3_BUCKET}/datasets/openimages/annotations/oidv6-train-annotations-bbox.csv"
        )
        train_annotations.write_csv(ann_cache)
    else:
        train_annotations = pl.read_csv(ann_cache)

    if not os.path.exists(cls_cache):
        print("Downloading class descriptions from S3...")
        class_descriptions = pl.read_csv(
            f"s3://{S3_BUCKET}/datasets/openimages/annotations/class-descriptions-boxable.csv",
            has_header=False,
        )
        class_descriptions.write_csv(cls_cache)
    else:
        class_descriptions = pl.read_csv(cls_cache)

    return train_annotations, class_descriptions


def get_image_set(train_annotations, class_descriptions, class_name):
    cls_id = class_descriptions.filter(
        pl.col("column_2") == class_name
    )["column_1"][0]
    return set(
        train_annotations.filter(pl.col("LabelName") == cls_id)["ImageID"].to_list()
    )


# OpenImages boxable class names that denote a person. Kept in sync with the wider
# recognition.PERSON_LABELS set used at pseudo-label time, restricted to names that
# actually exist as OpenImages boxable classes (Boy/Girl/Child/People are matched at
# labeling time but are not OpenImages boxable classes, so they can't be filtered here).
_OPENIMAGES_PERSON_CLASSES = ("Person", "Man", "Woman", "Boy", "Girl")


def get_valid_image_ids(s3_client, train_annotations, class_descriptions):
    """Get image IDs that have trains but no people and exist in S3."""
    train_images = get_image_set(train_annotations, class_descriptions, "Train")
    people_images: set[str] = set()
    for person_class in _OPENIMAGES_PERSON_CLASSES:
        try:
            people_images |= get_image_set(train_annotations, class_descriptions, person_class)
        except IndexError:
            # class not present in this class-descriptions file; skip it
            continue
    train_no_people = train_images - people_images

    # Check which are actually downloaded
    downloaded_ids = set()
    paginator = s3_client.get_paginator("list_objects_v2")
    print(f"Listing images in s3://{S3_BUCKET}/{SOURCE_IMAGES}...")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=SOURCE_IMAGES):
        for obj in page.get("Contents", []):
            image_id = obj["Key"].split("/")[-1].replace(".jpg", "")
            downloaded_ids.add(image_id)

    # Sort so the ordering is stable across processes. Python randomizes string
    # hashing per process (PYTHONHASHSEED), so a raw set-intersection list would
    # order image_ids differently each run -- which would break the per-index
    # seeding below (index i must map to the same image_id on a fresh run and a
    # resumed run for the reproducibility guarantee in generate() to hold).
    valid = sorted(downloaded_ids & train_no_people)
    print(f"Found {len(valid)} valid images (train, no people, downloaded)")
    return valid



# ---------------------------------------------------------------------------
# Data YAML generation
# ---------------------------------------------------------------------------

def create_data_yaml(synthetic_dir_name: str) -> str:
    """Create a YOLO data.yaml that references original + synthetic dirs."""
    yaml_content = f"""# Ablation dataset (original + {synthetic_dir_name})
path: .
train:
  - images/train_original
  - images/{synthetic_dir_name}
val: images/val_original
test: images/test_original

names:
"""
    for class_id, class_name in sorted(YOLO_CLASS_NAMES.items()):
        yaml_content += f"  {class_id}: {class_name}\n"
    return yaml_content


def copy_original_data(s3_client, dataset_prefix: str):
    """Copy val/test original data and train_original into the new dataset prefix."""
    # Directories to copy from source into the new dataset
    dirs_to_copy = [
        "images/train_original/",
        "labels/train_original/",
        "images/val_original/",
        "labels/val_original/",
        "images/test_original/",
        "labels/test_original/",
    ]

    for subdir in dirs_to_copy:
        src_prefix = SOURCE_DATASET + subdir
        dst_prefix = dataset_prefix + subdir

        # Check if destination already has files (skip if so)
        paginator = s3_client.get_paginator("list_objects_v2")
        existing = 0
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=dst_prefix, MaxKeys=1):
            existing += len(page.get("Contents", []))
        if existing > 0:
            print(f"  {subdir} already exists in destination, skipping copy")
            continue

        print(f"  Copying {subdir}...")
        count = 0
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=src_prefix):
            for obj in page.get("Contents", []):
                src_key = obj["Key"]
                filename = src_key[len(src_prefix):]
                dst_key = dst_prefix + filename
                s3_client.copy_object(
                    Bucket=S3_BUCKET,
                    CopySource={"Bucket": S3_BUCKET, "Key": src_key},
                    Key=dst_key,
                )
                count += 1
        print(f"    Copied {count} files")


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(args):
    s3_client = boto3.client("s3")
    rekognition = boto3.client("rekognition")

    # Load annotations and find target images
    train_annotations, class_descriptions = load_annotations_cached(s3_client)
    image_ids = get_valid_image_ids(s3_client, train_annotations, class_descriptions)

    # Output dataset layout
    dataset_prefix = args.dataset_prefix.rstrip("/") + "/"
    synthetic_dir = f"train_{args.output_suffix}"
    output_images_prefix = f"{dataset_prefix}images/{synthetic_dir}/"
    output_labels_prefix = f"{dataset_prefix}labels/{synthetic_dir}/"

    # Copy original data into the new dataset (idempotent)
    print(f"Setting up dataset at s3://{S3_BUCKET}/{dataset_prefix}")
    copy_original_data(s3_client, dataset_prefix)

    # Upload data.yaml
    data_yaml = create_data_yaml(synthetic_dir)
    yaml_key = f"{dataset_prefix}data.yaml"
    s3_client.put_object(
        Bucket=S3_BUCKET, Key=yaml_key, Body=data_yaml.encode("utf-8")
    )
    print(f"Uploaded {yaml_key}")

    # Resume cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"processed_{args.output_suffix}.json")
    cache = load_cache(cache_file)
    processed_set = set(cache["processed_ids"])
    failed_set = set(cache["failed_ids"])
    print(f"Resuming: {len(processed_set)} processed, {len(failed_set)} failed")

    # Load model
    pipeline = load_pipeline(model_parallel=args.model_parallel)
    resizer = ImageResizer(min_dim=DEFAULT_MIN_DIM)

    processed_count = 0
    for i, image_id in enumerate(tqdm(image_ids)):
        if image_id in processed_set or image_id in failed_set:
            continue

        try:
            # Sample augmentation parameters. Seed per-image on the absolute index
            # (like the diffusion seed below) so the values are reproducible and
            # independent of how many images a resume skipped -- a run started from
            # scratch and a resumed run produce identical augmentation for image i.
            rng = np.random.default_rng(SEED_BASE + i)
            gender = rng.choice(["male", "female"])
            time_of_day = rng.choice(
                ["day", "night", "dawn", "dusk"], p=[0.33, 0.33, 0.17, 0.17]
            )
            ambient_condition = rng.choice(
                ["normal", "dusty", "light fog", "very light rain"],
                p=[0.40, 0.30, 0.20, 0.10],
            )

            prompt = build_prompt(
                gender=gender,
                placement=args.placement,
                scene_augmentation=args.scene_augmentation,
                time_of_day=time_of_day,
                ambient_condition=ambient_condition,
            )

            image = read_image_from_s3(s3_client, S3_BUCKET, image_id, SOURCE_IMAGES)

            output_image = edit_image(
                pipeline,
                resizer.resize(image),
                prompt,
                seed=SEED_BASE + i,
            )
            resized = resizer.restore(output_image)

            # Detect people with Rekognition. Force RGB so a non-RGB source (e.g.
            # grayscale/CMYK/RGBA) still encodes cleanly as JPEG.
            buffer = BytesIO()
            resized.convert("RGB").save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()

            response = rekognition.detect_labels(
                Image={"Bytes": image_bytes},
                MaxLabels=10,
                MinConfidence=80,
                Features=["GENERAL_LABELS"],
            )

            person_boxes = extract_person_boxes(response["Labels"])
            deduped_boxes = dedupe_person_boxes(person_boxes)

            # Read existing label
            try:
                label_obj = s3_client.get_object(
                    Bucket=S3_BUCKET, Key=f"{SOURCE_LABELS}{image_id}.txt"
                )
                existing_label = label_obj["Body"].read().decode("utf-8")
            except s3_client.exceptions.NoSuchKey:
                print(f"Missing label for {image_id}, skipping")
                cache["failed_ids"].append(image_id)
                failed_set.add(image_id)
                save_cache(cache, cache_file)
                continue

            # Build combined label (existing train boxes + new person boxes)
            new_lines = []
            for box in deduped_boxes:
                x_center = box["bbox"]["Left"] + box["bbox"]["Width"] / 2
                y_center = box["bbox"]["Top"] + box["bbox"]["Height"] / 2
                width = box["bbox"]["Width"]
                height = box["bbox"]["Height"]
                new_lines.append(
                    f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

            combined_label = existing_label.strip()
            if new_lines:
                if combined_label:
                    combined_label += "\n"
                combined_label += "\n".join(new_lines)

            # Upload to S3
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=f"{output_images_prefix}{image_id}.jpg",
                Body=image_bytes,
            )
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=f"{output_labels_prefix}{image_id}.txt",
                Body=combined_label.encode("utf-8"),
            )

            print(f"Saved {image_id} with {len(deduped_boxes)} person boxes")
            cache["processed_ids"].append(image_id)
            processed_set.add(image_id)

            if i % 10 == 0:
                save_cache(cache, cache_file)

            processed_count += 1
            if args.max_images > 0 and processed_count >= args.max_images:
                print(f"Reached max_images={args.max_images}, stopping.")
                break

        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            cache["failed_ids"].append(image_id)
            failed_set.add(image_id)
            save_cache(cache, cache_file)
            continue

    save_cache(cache, cache_file)
    print(f"Done. Processed: {len(processed_set)}, Failed: {len(failed_set)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _str2bool(value):
    """Parse a boolean flag that may be bare (``--flag``) or valued (``--flag true``).

    SageMaker always emits hyperparameters as ``--key value`` (never a bare flag),
    passing booleans as ``"true"``/``"false"`` strings, while a local CLI run uses the
    bare flag. Both are supported: a bare flag yields ``True`` (via ``const``), and a
    value is interpreted here (an empty string also means ``True``).
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("", "1", "true", "yes", "y")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training images with Qwen Image Edit"
    )
    parser.add_argument(
        "--placement",
        choices=["hazardous", "background"],
        required=True,
        help="Where to place the synthetic person",
    )
    parser.add_argument(
        "--scene-augmentation",
        nargs="?",
        const=True,
        default=False,
        type=_str2bool,
        help="Include time-of-day and ambient condition variation in the prompt",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        required=True,
        help="S3 prefix for the output dataset (e.g. 'datasets/ablation_hazardous_scene/'). "
             "Original data will be copied here and synthetic data added alongside it.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        required=True,
        help="Suffix for synthetic subdirs (e.g. 'hazardous_scene' -> images/train_hazardous_scene/)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max images to process THIS run, not counting resume-skipped ones "
             "(0 = all)",
    )
    parser.add_argument(
        "--model-parallel",
        nargs="?",
        const=True,
        default=False,
        type=_str2bool,
        help="Shard the model across multiple GPUs for instances that can't fit "
             "it on one device (e.g. 4x A10G g5.12xlarge). Omit for a single "
             "large-memory GPU (H100/B200).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
