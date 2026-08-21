# Download OpenImages images for a set of class labels.
#
# Images are pulled from the public OpenImages mirror on AWS
# (s3://open-images-dataset), either to
# local disk or copied server-side into your own S3 bucket. Only images are
# fetched here; YOLO labels are produced downstream by create_yolo_dataset.py.
#
# See __main__ for example usage.
import argparse
import concurrent.futures
import io
import logging
import os
from typing import Dict, List, Optional, Set
import urllib3
import warnings

import boto3
import botocore
import pandas as pd
import requests
from tqdm import tqdm


# define a "public API" and somewhat manage "wild" imports
# (see http://xion.io/post/code/python-all-wild-imports.html)
__all__ = ["download_dataset", "download_dataset_to_s3"]

# Source bucket for OpenImages (public, no auth needed for reads)
OPENIMAGES_S3_BUCKET = "open-images-dataset"

# OpenImages annotation CSV locations. Bounding-box annotations are versioned by
# split: train uses the v6 bbox CSV, while validation/test and the class
# descriptions come from v5.
_OID_v6 = "https://storage.googleapis.com/openimages/v6/"
_OID_v5 = "https://storage.googleapis.com/openimages/v5/"

# Silence only the noisy "Connection pool is full" warnings that urllib3/boto
# emit under the ThreadPoolExecutor -- not all warnings, so real deprecation and
# pandas warnings still surface.
warnings.filterwarnings("ignore", message="Connection pool is full")

# ------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def _class_label_codes(
        class_labels: List[str],
        csv_dir: str = None,
) -> Dict:
    """
    Gets a dictionary that maps a list of OpenImages image class labels to their
    corresponding image class label codes.

    :param class_labels: image class labels for which we'll find corresponding
        OpenImages image class codes
    :param csv_dir: directory where we should look for the class descriptions
        CSV file, and if not present download it into there for future use
    :return: dictionary with the class labels mapped to their corresponding
        OpenImages image class codes
    """

    classes_csv = "class-descriptions-boxable.csv"

    if csv_dir is None:

        # get the class descriptions CSV from OpenImages and read into a DataFrame
        url = _OID_v5 + classes_csv
        response = requests.get(url, allow_redirects=True)
        if response.status_code != 200:
            raise ValueError(
                "Failed to get class descriptions information -- Invalid "
                f"response (status code: {response.status_code}) from {url}",
            )
        df_classes = pd.read_csv(io.BytesIO(response.content), header=None)

    else:

        # download the class descriptions CSV file to the specified directory if not present
        descriptions_csv_file_path = os.path.join(csv_dir, classes_csv)
        if not os.path.exists(descriptions_csv_file_path):

            # get the annotations CSV for the section
            url = _OID_v5 + classes_csv
            response = requests.get(url, allow_redirects=True)
            if response.status_code != 200:
                raise ValueError(
                    "Failed to get class descriptions information -- Invalid "
                    f"response (status code: {response.status_code}) from {url}",
                )
            with open(descriptions_csv_file_path, "wb") as descriptions_csv_file:
                descriptions_csv_file.write(response.content)

        df_classes = pd.read_csv(descriptions_csv_file_path, header=None)

    # build dictionary of class labels to OpenImages class codes
    labels_to_codes = {}
    for class_label in class_labels:
        labels_to_codes[class_label.lower()] = \
            df_classes.loc[df_classes[1] == class_label].values[0][0]

    # return the labels to OpenImages codes dictionary
    return labels_to_codes


# ------------------------------------------------------------------------------
def download_dataset(
        dest_dir: str,
        class_labels: List[str],
        exclusions_path: str = None,
        csv_dir: str = None,
        limit: int = None,
) -> Dict:
    """
    Downloads images to local disk for a specified list of OpenImages class labels.

    :param dest_dir: base directory under which the images will be stored
    :param class_labels: list of OpenImages class labels we'll download
    :param exclusions_path: path to file containing file IDs to exclude from the
        dataset (useful if there are files known to be problematic or invalid)
    :param csv_dir: directory where we should look for the class descriptions
        and annotations CSV files, if these files are not present from a previous
        usage then download these files into this directory for future use
    :param limit: the maximum number of images per label we should download
    :return: dictionary of class labels mapped to dictionaries specifying the
        corresponding images directory for the class
    """

    # make the metadata directory if it's specified and doesn't exist
    if csv_dir is not None:
        os.makedirs(csv_dir, exist_ok=True)

    # get the OpenImages image class codes for the specified class labels
    label_codes = _class_label_codes(class_labels, csv_dir)

    # build the images directory for each class label
    class_directories = {}
    for class_label in label_codes.keys():
        images_dir = os.path.join(dest_dir, class_label, "images")
        os.makedirs(images_dir, exist_ok=True)
        class_directories[class_label] = {"images_dir": images_dir}

    # get the IDs of questionable files marked for exclusion
    exclusion_ids = None
    if exclusions_path is not None:

        # read the file IDs from the exclusions file
        with open(exclusions_path, "r") as exclusions_file:
            exclusion_ids = set([line.rstrip('\n') for line in exclusions_file])

    # keep counts of the number of images downloaded for each label
    class_labels = list(label_codes.keys())
    label_download_counts = {label: 0 for label in class_labels}

    # OpenImages is already split into sections so we'll need to loop over each
    for split_section in ("train", "validation", "test"):

        # get a dictionary of class labels to GroupByDataFrames
        # containing bounding box info grouped by image IDs
        label_bbox_groups = _group_bounding_boxes(split_section, label_codes, exclusion_ids, csv_dir)

        for class_label in class_labels:

            # get the bounding boxes grouped by image and the collection of image IDs
            bbox_groups = label_bbox_groups[class_label]
            image_ids = bbox_groups.groups.keys()

            # limit the number of images we'll download, if specified
            if limit is not None:
                remaining = limit - label_download_counts[class_label]
                if remaining <= 0:
                    # this label is full; other labels in this split may not be
                    continue
                elif remaining < len(image_ids):
                    image_ids = list(image_ids)[0:remaining]

            # download the images
            _logger.info(
                f"Downloading {len(image_ids)} {split_section} images "
                f"for class \'{class_label}\'",
            )
            _download_images_by_id(
                image_ids,
                split_section,
                class_directories[class_label]["images_dir"],
            )

            # update the downloaded images count for this label
            label_download_counts[class_label] += len(image_ids)

    return class_directories


# ------------------------------------------------------------------------------
def download_dataset_to_s3(
        dest_s3_bucket: str,
        dest_s3_prefix: str,
        class_labels: List[str],
        exclusions_path: str = None,
        csv_dir: str = None,
        limit: int = None,
) -> Dict:
    """
    Downloads images directly from OpenImages S3 bucket to your own S3 bucket.
    No local disk storage required - pure S3-to-S3 copy.

    This is ideal for running on EC2 in the same region as your destination bucket
    to avoid data transfer costs and maximize throughput.

    :param dest_s3_bucket: destination S3 bucket name (e.g., "amzn-s3-demo-bucket")
    :param dest_s3_prefix: prefix/path in destination bucket (e.g., "openimages/person")
    :param class_labels: list of OpenImages class labels to download (e.g., ["Person"])
    :param exclusions_path: path to file containing image IDs to exclude
    :param csv_dir: local directory for caching CSV metadata files
    :param limit: maximum number of images per label to download
    :return: dictionary with class labels mapped to their S3 prefixes

    Usage:
        download_dataset_to_s3(
            dest_s3_bucket="amzn-s3-demo-bucket",
            dest_s3_prefix="datasets/openimages",
            class_labels=["Person"],
            csv_dir="/tmp/openimages_csv",
            limit=10000
        )
    """

    # Create local csv_dir for metadata if specified
    if csv_dir is not None:
        os.makedirs(csv_dir, exist_ok=True)

    # Get the OpenImages class codes for the specified labels
    label_codes = _class_label_codes(class_labels, csv_dir)

    # Build the S3 prefixes for each class label
    class_s3_paths = {}
    for class_label in label_codes.keys():
        class_s3_paths[class_label] = {
            "s3_prefix": f"{dest_s3_prefix}/{class_label}/images" if dest_s3_prefix else f"{class_label}/images",
        }

    # Get IDs of files to exclude
    exclusion_ids = None
    if exclusions_path is not None:
        with open(exclusions_path, "r") as exclusions_file:
            exclusion_ids = set([line.rstrip('\n') for line in exclusions_file])

    # Track download counts per label
    class_labels = list(label_codes.keys())
    label_download_counts = {label: 0 for label in class_labels}

    # Process each split section
    for split_section in ("train", "validation", "test"):

        label_bbox_groups = _group_bounding_boxes(split_section, label_codes, exclusion_ids, csv_dir)

        for class_label in class_labels:

            bbox_groups = label_bbox_groups[class_label]
            image_ids = list(bbox_groups.groups.keys())

            # Apply limit if specified
            if limit is not None:
                remaining = limit - label_download_counts[class_label]
                if remaining <= 0:
                    # this label is full; other labels in this split may not be
                    continue
                elif remaining < len(image_ids):
                    image_ids = image_ids[0:remaining]

            if len(image_ids) == 0:
                continue

            _logger.info(
                f"Copying {len(image_ids)} {split_section} images "
                f"for class '{class_label}' to s3://{dest_s3_bucket}/{class_s3_paths[class_label]['s3_prefix']}/",
            )

            # Use S3-to-S3 copy
            _download_images_by_id(
                image_ids,
                split_section,
                images_directory=None,  # Not used in S3-to-S3 mode
                dest_s3_bucket=dest_s3_bucket,
                dest_s3_prefix=f"{class_s3_paths[class_label]['s3_prefix']}/{split_section}",
            )

            label_download_counts[class_label] += len(image_ids)

    _logger.info(f"Download complete. Total images per class: {label_download_counts}")
    return class_s3_paths


# ------------------------------------------------------------------------------
def _download_images_by_id(
        image_ids: List[str],
        section: str,
        images_directory: str,
        dest_s3_bucket: Optional[str] = None,
        dest_s3_prefix: Optional[str] = None,
):
    """
    Downloads a collection of images from OpenImages dataset.

    Can either download to local disk OR copy directly to another S3 bucket.

    :param image_ids: list of image IDs to download
    :param section: split section (train, validation, or test) where the image
        should be found
    :param images_directory: destination directory where the image files are to
        be written (used for local download mode)
    :param dest_s3_bucket: destination S3 bucket for S3-to-S3 copy (optional)
    :param dest_s3_prefix: destination S3 prefix/path for S3-to-S3 copy (optional)
    """

    # Determine if we're doing S3-to-S3 copy or local download
    s3_to_s3_mode = dest_s3_bucket is not None

    if s3_to_s3_mode:
        # For S3-to-S3, we need a client that can write to the destination bucket
        # (assumes IAM role or credentials are configured)
        s3_client = boto3.client('s3')
    else:
        # For local download, use unsigned requests to read from public bucket
        s3_client = boto3.client(
            's3',
            config=botocore.config.Config(signature_version=botocore.UNSIGNED),
        )

    # create an iterable list of function arguments
    # that we'll map to the download function
    download_args_list = []
    for image_id in image_ids:
        image_file_name = image_id + ".jpg"
        source_key = section + "/" + image_file_name

        if s3_to_s3_mode:
            dest_key = f"{dest_s3_prefix}/{image_file_name}" if dest_s3_prefix else image_file_name
            download_args = {
                "s3_client": s3_client,
                "source_bucket": OPENIMAGES_S3_BUCKET,
                "source_key": source_key,
                "dest_bucket": dest_s3_bucket,
                "dest_key": dest_key,
                "s3_to_s3": True,
            }
        else:
            download_args = {
                "s3_client": s3_client,
                "image_file_object_path": source_key,
                "dest_file_path": os.path.join(images_directory, image_file_name),
                "s3_to_s3": False,
            }
        download_args_list.append(download_args)

    # use a ThreadPoolExecutor to download the images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:

        # use the executor to map the download function to the iterable of arguments
        list(tqdm(executor.map(_download_single_image, download_args_list),
                  total=len(download_args_list)))


# ------------------------------------------------------------------------------
def _get_annotations_csv(
        split_section: str,
) -> requests.Response:
    """
    Requests the annotations CSV for a split section.

    :param split_section:
    :return: a requests.Response object containing the CSV payload
    """

    # get the annotations CSV for the section
    # v6 has the most recent bbox annotations that are publicly accessible
    if split_section == "train":
        url = _OID_v6 + "oidv6-train-annotations-bbox.csv"
    else:
        url = _OID_v5 + split_section + "-annotations-bbox.csv"
    response = requests.get(url, allow_redirects=True)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to get bounding box information for split section {split_section} "
            f"-- Invalid response (status code: {response.status_code}) from {url}",
        )

    return response


# ------------------------------------------------------------------------------
def _group_bounding_boxes(
        section: str,
        label_codes: Dict,
        exclusion_ids: Set[str],
        csv_dir: str = None,
) -> Dict:
    """
    Gets a pandas DataFrameGroupBy object containing bounding boxes for an image
    class grouped by image ID.

    :param section: the relevant split section, "train", "validation", or "test"
    :param label_codes: dictionary with class labels mapped to the
        corresponding OpenImages-specific code of the image class
    :param exclusion_ids: file IDs that should be excluded
    :param csv_dir
    :return: DataFrameGroupBy object with bounding box columns grouped by image IDs
    """

    if csv_dir is None:

        # get the annotations CSV for the section
        response = _get_annotations_csv(section)

        # read the CSV into a pandas DataFrame
        df_images = pd.read_csv(io.BytesIO(response.content))

    else:

        # download the annotations CSV file to the specified directory if not present
        if section == "train":
            csv_filename = "oidv6-train-annotations-bbox.csv"
        else:
            csv_filename = section + "-annotations-bbox.csv"
        bbox_csv_file_path = os.path.join(csv_dir, csv_filename)
        if not os.path.exists(bbox_csv_file_path):
            # get the annotations CSV for the section
            response = _get_annotations_csv(section)
            with open(bbox_csv_file_path, "wb") as annotations_file:
                annotations_file.write(response.content)

        # read the CSV into a pandas DataFrame
        df_images = pd.read_csv(bbox_csv_file_path)

    # remove any rows which are identified to be excluded
    if exclusion_ids and (len(exclusion_ids) > 0):
        df_images = df_images[~df_images["ImageID"].isin(exclusion_ids)]

    # filter out images that are occluded, truncated, group, depiction, inside, etc.
    for reject_field in ("IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"):
        df_images = df_images[df_images[reject_field] == 0]

    # drop the columns we won't need, keeping only
    # the image ID, label name and bounding box columns
    unnecessary_columns = [
        "IsOccluded",
        "IsTruncated",
        "IsGroupOf",
        "IsDepiction",
        "IsInside",
        "Source",
        "Confidence",
    ]
    df_images = df_images.drop(columns=unnecessary_columns)

    # create a dictionary and populate it with class labels mapped to
    # GroupByDataFrame objects with bounding boxes grouped by image ID
    labels_to_bounding_box_groups = {}
    for class_label, class_code in label_codes.items():

        # filter the DataFrame down to just the images for the class label
        # (.copy() so the subsequent column drop is on an owned frame, not a view)
        df_label_images = df_images[df_images["LabelName"] == class_code].copy()

        # drop the label name column since it's no longer needed
        df_label_images = df_label_images.drop(columns=["LabelName"])

        # map the class label to a GroupBy object with each
        # group's row containing the bounding box columns
        labels_to_bounding_box_groups[class_label] = \
            df_label_images.groupby("ImageID")

    # return the dictionary we've created
    return labels_to_bounding_box_groups


# ------------------------------------------------------------------------------
def _download_single_image(arguments: Dict):
    """
    Downloads and saves an image file from the OpenImages dataset.

    Supports two modes:
    1. Local download: downloads from S3 to local filesystem
    2. S3-to-S3 copy: copies directly between S3 buckets (no local disk needed)

    :param arguments: dictionary containing the following arguments:
        For local download mode:
            "s3_client": an S3 client object
            "image_file_object_path": the S3 object path corresponding to the image
            "dest_file_path": destination path where the image file should be written
            "s3_to_s3": False
        For S3-to-S3 mode:
            "s3_client": an S3 client object (with write permissions to dest bucket)
            "source_bucket": source S3 bucket name
            "source_key": source S3 object key
            "dest_bucket": destination S3 bucket name
            "dest_key": destination S3 object key
            "s3_to_s3": True
    """

    try:
        if arguments.get("s3_to_s3", False):
            # S3-to-S3 copy mode
            copy_source = {
                'Bucket': arguments["source_bucket"],
                'Key': arguments["source_key"]
            }
            arguments["s3_client"].copy(
                copy_source,
                arguments["dest_bucket"],
                arguments["dest_key"],
            )
        else:
            # Local download mode
            with open(arguments["dest_file_path"], "wb") as dest_file:
                arguments["s3_client"].download_fileobj(
                    OPENIMAGES_S3_BUCKET,
                    arguments["image_file_object_path"],
                    dest_file,
                )

    except urllib3.exceptions.ProtocolError as error:
        key = arguments.get("source_key") or arguments.get("image_file_object_path")
        _logger.warning(f"Unable to download/copy image {key}: {error} -- skipping")
    except Exception as error:
        key = arguments.get("source_key") or arguments.get("image_file_object_path")
        _logger.warning(f"Error processing image {key}: {error} -- skipping")


# ------------------------------------------------------------------------------
def _parse_command_line():

    # parse the command line arguments
    args_parser = argparse.ArgumentParser(
        description="Download OpenImages images to local disk or copy them "
                    "server-side into your own S3 bucket.",
    )
    args_parser.add_argument(
        "--base-dir",
        type=str,
        required=False,
        help="path to the base output directory (for local download)",
    )
    args_parser.add_argument(
        "--labels",
        type=str,
        required=True,
        nargs='+',
        help="object class to be fetched from OpenImages",
    )
    args_parser.add_argument(
        "--exclusions",
        type=str,
        required=False,
        help="path to file containing file IDs (one per line) to exclude from "
             "the final dataset",
    )
    args_parser.add_argument(
        "--csv-dir",
        type=str,
        required=False,
        help="path to a directory where CSV files for the OpenImages dataset "
             "metadata (annotations, descriptions, etc.) should be read and/or "
             "downloaded into for later use",
    )
    args_parser.add_argument(
        "--limit",
        type=int,
        required=False,
        help="maximum number of images to download per image class/label",
    )
    # S3-to-S3 copy arguments
    args_parser.add_argument(
        "--s3-bucket",
        dest="s3_bucket",
        type=str,
        required=False,
        help="destination S3 bucket for S3-to-S3 copy (e.g., 'amzn-s3-demo-bucket'). "
             "Falls back to $SDA_S3_BUCKET when neither --s3-bucket nor --base-dir "
             "is given.",
    )
    args_parser.add_argument(
        "--s3-prefix",
        dest="dest_s3_prefix",
        type=str,
        required=False,
        default="datasets/openimages",
        help="destination S3 prefix/path (default: 'datasets/openimages', matching "
             "create_yolo_dataset.py's --source-prefix default)",
    )
    return vars(args_parser.parse_args())


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Usage RECOMMENDED (S3-to-S3 copy - no local disk needed):
    $ python download_openimages.py --s3-bucket amzn-s3-demo-bucket \
          --s3-prefix datasets/openimages \
          --labels Person --csv-dir /tmp/csv_cache --limit 10000
    Usage (local download):
    $ python download_openimages.py --base-dir /data/datasets/openimages \
          --labels Person --csv-dir /data/datasets/openimages
    """

    args = _parse_command_line()

    # The S3 bucket is opt-in via an explicit flag so it also selects S3 mode; only
    # when neither mode flag is given do we fall back to $SDA_S3_BUCKET (S3 mode),
    # mirroring qwen_image_edit/config.py (kept inline: this script runs outside that
    # package). A --base-dir run stays local even if SDA_S3_BUCKET happens to be set.
    dest_s3_bucket = args["s3_bucket"]

    # Dispatch based on the mode implied by the arguments
    if dest_s3_bucket is not None:
        # S3-to-S3 mode (explicit --s3-bucket)
        download_dataset_to_s3(
            dest_s3_bucket,
            args["dest_s3_prefix"],
            args["labels"],
            args["exclusions"],
            args["csv_dir"],
            args["limit"],
        )
    elif args["base_dir"] is not None:
        # Local download mode
        download_dataset(
            args["base_dir"],
            args["labels"],
            args["exclusions"],
            args["csv_dir"],
            args["limit"],
        )
    elif os.environ.get("SDA_S3_BUCKET"):
        # No mode flag given, but a destination bucket is configured via the env
        # var: default to S3-to-S3 mode using it (env can't supply a local path).
        download_dataset_to_s3(
            os.environ["SDA_S3_BUCKET"],
            args["dest_s3_prefix"],
            args["labels"],
            args["exclusions"],
            args["csv_dir"],
            args["limit"],
        )
    else:
        raise SystemExit(
            "No S3 bucket configured. Set SDA_S3_BUCKET, or pass --s3-bucket "
            "(S3 mode) or --base-dir (local mode)."
        )
