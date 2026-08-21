#!/usr/bin/env bash
# Upload this repo to S3 so it can be pulled onto SageMaker notebook instances.
#
# Usage:
#   export SDA_S3_BUCKET=amzn-s3-demo-bucket
#   ./scripts/upload_repo.sh
#
# Then on the notebook instance:
#   aws s3 sync s3://$SDA_S3_BUCKET/repo/synthetic_data_augmentation/ ~/SageMaker/synthetic_data_augmentation/
#   cd ~/SageMaker/synthetic_data_augmentation
#   Follow the README GPU setup instructions.

set -euo pipefail

BUCKET="${SDA_S3_BUCKET:?Set SDA_S3_BUCKET to your bucket name}"
PREFIX="repo/synthetic_data_augmentation"

# Must be run from within the git work tree.
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "Error: run this from inside the git repo." >&2
    exit 1
}
cd "$(git rev-parse --show-toplevel)"

# Exclude untracked paths by deriving patterns from git rather than maintaining
# a hardcoded list. Tracked files are uploaded from the working tree, including
# any modifications. `git ls-files --others --directory` lists untracked and
# ignored paths alike (it does not apply ignore rules without
# --exclude-standard), with directories collapsed. Turn each into an
# `aws s3 sync --exclude` pattern (a trailing-slash dir becomes "dir/*").
EXCLUDES=(--exclude ".git/*")
while IFS= read -r path; do
    [ -z "$path" ] && continue
    case "$path" in
        */) EXCLUDES+=(--exclude "${path}*") ;;
        *)  EXCLUDES+=(--exclude "$path") ;;
    esac
done < <(git ls-files --others --directory)

echo "Uploading repo to s3://${BUCKET}/${PREFIX}/"
echo "Applying ${#EXCLUDES[@]} exclusion pattern(s) for .git and untracked paths; tracked working-tree modifications are uploaded."

aws s3 sync . "s3://${BUCKET}/${PREFIX}/" "${EXCLUDES[@]}"

echo "Done. Pull onto notebook instance with:"
echo "  aws s3 sync s3://${BUCKET}/${PREFIX}/ ~/SageMaker/synthetic_data_augmentation/"
