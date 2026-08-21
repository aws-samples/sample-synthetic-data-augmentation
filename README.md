# Synthetic Data Augmentation

A project for augmenting object detection datasets with synthetic data using AI image editing. It inserts synthetic people into real images of trains (locomotives) to study whether the added data improves person and train detection.

## Overview

This project explores using generative AI (e.g., Qwen image editing) to create synthetic training data for YOLO object detection models. The workflow:

1. Download images from OpenImages dataset containing trains (locomotives)
2. Use AI image editing to add synthetic people to train images
3. Train YOLO models on combined original + synthetic data
4. Evaluate if synthetic augmentation improves detection performance

**Note:** This project is a research/experimentation pipeline, not
production-ready software. Review and harden before adapting it for
a production system.

## Prerequisites

Before you start, you'll need:

- **Python 3.11** (`>=3.11,<3.12`) is required. Install `uv` using its
  [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
- **An AWS account with credentials configured** (`aws configure`, environment
  variables, or an IAM role). The pipeline reads and writes S3 throughout.
- **A GPU with 40GB+ VRAM** for the image-editing step (single H100/B200), or a
  multi-GPU instance such as `ml.g5.12xlarge` (4× A10G) for the sharded path.
  Everything here is set up to run on **Amazon SageMaker AI**.
- **SageMaker AI GPU service quota.** New accounts have a quota of **0** for GPU
  training instances (e.g. `ml.g5.12xlarge`); request an increase in the AWS
  Service Quotas console before launching jobs, or they'll fail / sit pending.
- **Amazon Rekognition access** (`rekognition:DetectLabels`) — a billable AWS
  service used to auto-label the synthetic people.
- **Hugging Face access** to download the `Qwen/Qwen-Image-Edit-2509` model
  weights (downloaded automatically on first run).

Costs are incurred for SageMaker AI GPU time, Rekognition calls, and S3 storage.

## Project Structure

```text
├── data_prep/                    # Data preparation scripts
│   ├── download_openimages.py    # Download images from the public OpenImages Amazon S3 mirror
│   └── create_yolo_dataset.py    # Convert to YOLO format
├── qwen_image_edit/              # Synthetic data generation
│   ├── config.py                 # SDA_S3_BUCKET config
│   ├── prompts.py                # Prompt templates
│   ├── recognition.py            # Amazon Rekognition pseudo-labeling
│   ├── s3_io.py                  # Amazon S3 image I/O, resizing, resume cache
│   ├── model.py                  # Qwen pipeline loading (single- or multi-GPU) + inference
│   ├── generate_synthetic.py     # Batch generation CLI
│   ├── launch_generation.py      # Launch generation as an Amazon SageMaker AI job
│   ├── requirements.txt          # Generation job dependencies (installed by the DLC)
│   └── generate.ipynb            # Demo notebook (single GPU or multi-GPU via a flag)
├── yolo_training/                # Model training
│   ├── train.py                  # Amazon SageMaker AI training script (supports --seed)
│   ├── launch_training.py        # Launch Amazon SageMaker AI training jobs
│   └── requirements.txt          # Training dependencies
├── evaluation/                   # Evaluation & analysis
│   ├── extract_per_class.py      # Per-class mAP breakdown (person vs train)
│   ├── run_eval.py               # Run YOLO validation with plots
│   ├── compare_results.py        # Side-by-side visual comparison (2- or 3-way)
│   ├── find_missed_detections.py # Find/annotate missed detections
│   ├── download_testdata.py      # Download test set from Amazon S3
│   └── README.md                 # Evaluation workflow docs
├── scripts/                      # GPU generation environment and repo upload helper
│   ├── upload_repo.sh            # Sync the repo to Amazon S3 for notebook instances
│   ├── pyproject.toml            # Notebook-instance generation dependencies
│   └── uv.lock                   # Locked GPU generation dependencies
├── .gitignore                    # Ignores datasets/, caches, model weights, etc.
├── .python-version               # Pins Python to 3.11 (for pyenv/uv)
├── CODE_OF_CONDUCT.md            # Community code of conduct
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # MIT-0 license
├── README.md                     # Project overview, setup, and full usage walkthrough
├── pyproject.toml                # Root project dependencies (uv-managed)
└── uv.lock                       # Locked dependency versions for the root pyproject.toml
```

## Classes

- `person` (class 0): Person, Man, Woman from OpenImages
- `train` (class 1): Train (locomotive) from OpenImages

## Installation

From the repository root, choose the environment for the task you plan to run.
You do not need to install both environments.

For data preparation, SageMaker job launchers, YOLO training, and evaluation,
install the general workflow dependencies from the root lockfile:

```bash
uv sync --frozen
```

For direct GPU generation with the CLI or notebook, install only the separate
locked environment in `scripts/`:

```bash
uv sync --frozen --project scripts
```

The direct-generation environment targets classic SageMaker AI Notebook
Instances running Linux x86_64 with an NVIDIA GPU. Verify that PyTorch can see
CUDA:

```bash
scripts/.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
```

To use this environment in Jupyter, register it and then select **Python
(synthetic generation)** in the notebook's kernel picker. The locked environment
already includes `ipykernel`:

```bash
scripts/.venv/bin/python -m ipykernel install --user --name synthetic-generation --display-name "Python (synthetic generation)"
```

## Configuration

Most scripts read the destination Amazon S3 bucket from the `SDA_S3_BUCKET` environment
variable:

```bash
export SDA_S3_BUCKET=amzn-s3-demo-bucket
```

Every script that takes a bucket uses the same `--s3-bucket` flag, which defaults
to `SDA_S3_BUCKET` (`download_openimages.py`, `create_yolo_dataset.py`,
`download_testdata.py`, and `launch_generation.py`). The one exception is
`generate_synthetic.py`, which reads `SDA_S3_BUCKET` directly with no CLI override
because it's the batch entry point — set it via `launch_generation.py --s3-bucket`,
which passes it through as the job's `SDA_S3_BUCKET`.

## Usage

### 1. Download OpenImages Data

#### Prerequisites: Create an S3 Bucket

The download script pulls images from the public OpenImages mirror on AWS
(`s3://open-images-dataset`), either to local disk or copied server-side into your own bucket. For the S3 workflow,
create a bucket first:

```bash
# Create a bucket (name must be globally unique)
aws s3 mb s3://amzn-s3-demo-bucket --region us-west-2

# Verify it was created
aws s3 ls | grep amzn-s3-demo-bucket
```

#### Download Images

```bash
# Download to S3
python data_prep/download_openimages.py \
    --labels Train \
    --s3-bucket amzn-s3-demo-bucket \
    --s3-prefix datasets/openimages \
    --csv-dir /tmp/openimages_csv
```

The script requires AWS credentials configured (via `aws configure`, environment variables, or IAM role) with `s3:PutObject` permission on your destination bucket.

This copies **images only** (no labels) for **all three OpenImages splits** —
train, validation, and test — into sibling prefixes:

```
s3://amzn-s3-demo-bucket/datasets/openimages/train/images/train/
s3://amzn-s3-demo-bucket/datasets/openimages/train/images/validation/
s3://amzn-s3-demo-bucket/datasets/openimages/train/images/test/
```

(The top-level `train/` comes from the `Train` class label, lowercased; the final
path component is the split.) Step 3 reads all three splits from here to build the
YOLO train/val/test sets, so don't skip the validation/test images. The OpenImages
annotation CSVs are cached locally under `--csv-dir`, not uploaded — the next step
stages them in Amazon S3.

### 2. Stage Annotation CSVs

Both `create_yolo_dataset.py` (below) and `generate_synthetic.py` (step 4) read the
OpenImages annotation CSVs from `s3://<bucket>/datasets/openimages/annotations/`.
Copy the four CSVs the download step cached locally into that prefix:

```bash
aws s3 cp /tmp/openimages_csv/oidv6-train-annotations-bbox.csv   s3://amzn-s3-demo-bucket/datasets/openimages/annotations/
aws s3 cp /tmp/openimages_csv/validation-annotations-bbox.csv    s3://amzn-s3-demo-bucket/datasets/openimages/annotations/
aws s3 cp /tmp/openimages_csv/test-annotations-bbox.csv          s3://amzn-s3-demo-bucket/datasets/openimages/annotations/
aws s3 cp /tmp/openimages_csv/class-descriptions-boxable.csv     s3://amzn-s3-demo-bucket/datasets/openimages/annotations/
```

### 3. Create YOLO Dataset

Reorganizes the downloaded images into the directory structure expected by YOLO
(`images/` and `labels/` folders with `data.yaml` / `data-with-synthetic.yaml`
configs), reading the staged annotation CSVs to generate the labels.

```bash
python data_prep/create_yolo_dataset.py \
    --source-prefix datasets/openimages/train/images/train/ \
    --output-prefix datasets/openimages_subset/
```

`--output-prefix` (default `datasets/openimages_subset/`) is where the generated
dataset lands; the synthetic-generation and training steps below consume that same
prefix, so keep it consistent if you change it.

### 4. Generate Synthetic Data

Add synthetic people to the train images with Qwen-Image-Edit-2509 and
pseudo-label them with Amazon Rekognition. On a **single large-memory GPU
(H100/B200)** the model fits in bf16 with no device map; on a **multi-GPU
instance** that can't fit it on one device (e.g. 4× A10G `g5.12xlarge`), the
transformer is hand-sharded across GPUs. Both are the same code path behind the
`model_parallel` flag in `model.py`:

- Interactive walkthrough: `qwen_image_edit/generate.ipynb` (set
  `MODEL_PARALLEL = True` for the multi-GPU case). Launch Jupyter from **inside
  `qwen_image_edit/`** — the notebook imports its sibling modules by bare name
  (`from config import ...`), so the working directory must be that folder.
- Unattended batch run on the current GPU box (add `--model-parallel` on a
  multi-GPU instance):

  ```bash
  scripts/.venv/bin/python qwen_image_edit/generate_synthetic.py \
      --placement hazardous \
      --dataset-prefix datasets/ablation_hazardous \
      --output-suffix hazardous
  ```

- Programmatic Amazon SageMaker AI job (no notebook needed) — launches the same script on a
  GPU instance via a PyTorch estimator. `--model-parallel` is on by default for the
  `ml.g5.12xlarge` (4× A10G) default instance:

  ```bash
  python qwen_image_edit/launch_generation.py \
      --s3-bucket amzn-s3-demo-bucket \
      --role arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/YourSageMakerRole \
      --placement hazardous \
      --dataset-prefix datasets/ablation_hazardous \
      --output-suffix hazardous
  ```

  The execution role needs Amazon S3 read/write on the bucket and
  `rekognition:DetectLabels`. Generation dependencies are installed in the job from
  `qwen_image_edit/requirements.txt`.

The batch script reads its source images from the hardcoded prefix
`datasets/openimages_subset/`, so step 3 must have used the default
`--output-prefix`.

**What this step produces.** Under the `--dataset-prefix` you pass, the script
writes a *complete, ready-to-train dataset*: it copies the original
train/val/test images and labels from `datasets/openimages_subset/` into that
prefix, adds the new synthetic images under `images/train_<output-suffix>/` (plus
matching labels), and writes its own `data.yaml` whose training set is
`train_original` **+** `train_<output-suffix>`. That generated `data.yaml` is the
one you train the "with synthetic" model on in step 5 — you do **not** reuse
`openimages_subset/data-with-synthetic.yaml` for this arm (its `train_synthetic`
folder is an empty placeholder created in step 3).

**Ablation conditions.** An *ablation* here means training the model on different
synthetic-data variants to isolate what actually helps. Two prompt knobs define
the variants; run this step once per condition, each with its own
`--dataset-prefix` / `--output-suffix`:

- `--placement {hazardous,background}` — where the synthetic person is placed.
  `hazardous` puts them in a dangerous spot (on the tracks, on top of the train)
  — the safety-detection scenario this project targets; `background` places them
  far from the train.
- `--scene-augmentation` (flag) — when set, folds a sampled time-of-day
  (day/night/dawn/dusk) and ambient condition (normal/dusty/light fog/very light
  rain) into the prompt; omit it for plain insertion.

The multi-GPU `g5.12xlarge` path is handled by the `model_parallel` flag above;
`generate.ipynb` covers both single- and multi-GPU cases.

### 5. Train YOLO Model

Train two models — a **baseline** on the original images only, and a
**synthetic-augmented** model — then compare them in step 6.

`--dataset-yaml` takes a **bare filename** resolved at the root of `--s3-data`
inside the Amazon SageMaker AI container, so each arm points `--s3-data` at the prefix that
holds the matching `data.yaml`:

- **Baseline** → the step-3 prefix (`datasets/openimages_subset/`), using its
  `data.yaml` (original images only).
- **Synthetic-augmented** → the step-4 `--dataset-prefix`
  (`datasets/ablation_hazardous/`), using the `data.yaml` that step 4 generated
  there (original + synthetic). This is the key wiring: the augmented model trains
  from step 4's output prefix, not from `openimages_subset/`.

```bash
# Baseline training (original images only) — from the step-3 prefix
python yolo_training/launch_training.py \
    --s3-data s3://amzn-s3-demo-bucket/datasets/openimages_subset/ \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/YourSageMakerRole \
    --dataset-yaml data.yaml \
    --epochs 100

# Training with synthetic augmentation — from the step-4 --dataset-prefix,
# using the data.yaml that step 4 generated there
python yolo_training/launch_training.py \
    --s3-data s3://amzn-s3-demo-bucket/datasets/ablation_hazardous/ \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/YourSageMakerRole \
    --dataset-yaml data.yaml \
    --epochs 100
```

> **Note on `data-with-synthetic.yaml`:** step 3 also writes a
> `data-with-synthetic.yaml` under `openimages_subset/`, but its `train_synthetic`
> directory is an empty placeholder. It's only useful if you populate that
> directory yourself; the batch generator in step 4 instead produces a complete
> augmented dataset (with its own `data.yaml`) under `--dataset-prefix`, which is
> what the command above trains on.

#### Setting Up a SageMaker AI Execution Role

The `--role` you pass to `launch_training.py` / `launch_generation.py` is the
**SageMaker execution role** — the role SageMaker assumes to run the training or
generation container. It only needs the permissions the workload actually uses:
S3 access to your bucket and the public OpenImages bucket, `rekognition:DetectLabels`
(generation only), plus the ECR pull and CloudWatch Logs write that let the job run.
Attach a scoped inline policy instead of `AmazonSageMakerFullAccess` /
`AmazonS3FullAccess`. Replace `amzn-s3-demo-bucket` with your bucket name.

You can create the role via the AWS Console or CLI:

**Option 1: AWS Console**
1. Go to Amazon IAM → Roles → Create role
2. Select "SageMaker" as the trusted service
3. Skip the AWS managed policies. Finish creating the role, then open it and add an
   **inline policy** using the JSON in Option 2 below (IAM → the role → Add
   permissions → Create inline policy → JSON).
4. Name the role (e.g., `SageMakerTrainingRole`) and create it
5. Copy the Role ARN for use with `--role`

**Option 2: AWS CLI**

```bash
# Create the trust policy file
cat > trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role
aws iam create-role \
    --role-name SageMakerTrainingRole \
    --assume-role-policy-document file://trust-policy.json

# Scoped execution-role policy: only what the workload uses.
cat > execution-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "UserBucketObjects",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::amzn-s3-demo-bucket/*"
    },
    {
      "Sid": "UserBucketList",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::amzn-s3-demo-bucket"
    },
    {
      "Sid": "OpenImagesRead",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::open-images-dataset",
        "arn:aws:s3:::open-images-dataset/*"
      ]
    },
    {
      "Sid": "SageMakerDefaultBucket",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::sagemaker-*", "arn:aws:s3:::sagemaker-*/*"]
    },
    {
      "Sid": "Rekognition",
      "Effect": "Allow",
      "Action": "rekognition:DetectLabels",
      "Resource": "*"
    },
    {
      "Sid": "EcrPull",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Sid": "Logs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "*"
    }
  ]
}
EOF

# Attach the scoped policy inline (no AmazonSageMakerFullAccess / AmazonS3FullAccess)
aws iam put-role-policy \
    --role-name SageMakerTrainingRole \
    --policy-name SyntheticDataExecutionPolicy \
    --policy-document file://execution-policy.json

# Get the role ARN
aws iam get-role --role-name SageMakerTrainingRole --query 'Role.Arn' --output text
```

> **Caller permissions (separate from the execution role).** The identity that
> *runs* `launch_training.py` / `launch_generation.py` — your local profile, or a
> notebook/EC2 instance role — is not the execution role above. It calls
> `CreateTrainingJob` and passes the execution role to SageMaker, so it needs
> `sagemaker:CreateTrainingJob`, `sagemaker:DescribeTrainingJob`, S3 access to your
> bucket, and `iam:PassRole` scoped to the execution-role ARN. Do **not** add these
> actions to the execution role.

### 6. Evaluate

Compare the baseline and synthetic-augmented models to see whether synthetic data
improved detection. The evaluation scripts live in `evaluation/`; see
[`evaluation/README.md`](evaluation/README.md) for the full workflow. In brief:

```bash
# Download the held-out test set and write val_dataset.yaml
python evaluation/download_testdata.py

# Run validation for BOTH trained models (compare_results.py needs both outputs)
python evaluation/run_eval.py --model-path ./models/best_baseline.pt \
    --data-yaml ./test_data/val_dataset.yaml
python evaluation/run_eval.py --model-path ./models/best_synthetic.pt \
    --data-yaml ./test_data/val_dataset.yaml

# Per-class mAP breakdown across models, then a side-by-side comparison
python evaluation/extract_per_class.py --data-yaml ./test_data/val_dataset.yaml \
    --models baseline:./models/best_baseline.pt,synthetic:./models/best_synthetic.pt
python evaluation/compare_results.py
```

## External Dependencies (Datasets and Models)
This sample code depends on and may incorporate or retrieve a number of third-party software packages (such as open source packages) at install-time or build-time or run-time ("External Dependencies"). The External Dependencies are subject to license terms that you must accept in order to use this package. If you do not accept all of the applicable license terms, you should not use this sample code. We recommend that you consult your company’s open source approval policy before proceeding.

Provided below is a list of External Dependencies and the applicable license identification as indicated by the documentation associated with the External Dependencies as of Amazon's most recent review.

THIS INFORMATION IS PROVIDED FOR CONVENIENCE ONLY. AMAZON DOES NOT PROMISE THAT THE LIST OR THE APPLICABLE TERMS AND CONDITIONS ARE COMPLETE, ACCURATE, OR UP-TO-DATE, AND AMAZON WILL HAVE NO LIABILITY FOR ANY INACCURACIES. YOU SHOULD CONSULT THE DOWNLOAD SITES FOR THE EXTERNAL DEPENDENCIES FOR THE MOST COMPLETE AND UP-TO-DATE LICENSING INFORMATION.

YOUR USE OF THE EXTERNAL DEPENDENCIES IS AT YOUR SOLE RISK. IN NO EVENT WILL AMAZON BE LIABLE FOR ANY DAMAGES, INCLUDING WITHOUT LIMITATION ANY DIRECT, INDIRECT, CONSEQUENTIAL, SPECIAL, INCIDENTAL, OR PUNITIVE DAMAGES (INCLUDING FOR ANY LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, OR COMPUTER FAILURE OR MALFUNCTION) ARISING FROM OR RELATING TO THE EXTERNAL DEPENDENCIES, HOWEVER CAUSED AND REGARDLESS OF THE THEORY OF LIABILITY, EVEN IF AMAZON HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. THESE LIMITATIONS AND DISCLAIMERS APPLY EXCEPT TO THE EXTENT PROHIBITED BY APPLICABLE LAW.

### OpenImages
Source dataset for train and person images.

**License:** The OpenImages annotations (bounding boxes, image-level labels)
are licensed by Google under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The images themselves are individually licensed under
[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) (sourced from Flickr);
Google notes it cannot guarantee the license status of every image, so verify
the license for any image you rely on individually. Both licenses permit
commercial use and modification (including the synthetic editing this
pipeline performs) but require attribution if the dataset, derived models, or
generated images are redistributed or published. See the
[OpenImages license page](https://storage.googleapis.com/openimages/web/factsfigures.html)
for details.

### Qwen-Image-Edit-2509
Image editing model used to add synthetic people to train images (see
`qwen_image_edit/`).

**License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), per the
[model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2509). Permissive —
no copyleft/open-sourcing obligations, and commercial use, modification, and
redistribution are allowed with attribution and inclusion of the license
notice.

### YOLO (Ultralytics)
Object detection model used for training and evaluation (`yolo11n.pt` by default,
see `yolo_training/`).

**License:** Ultralytics YOLO11 is dual-licensed under
[AGPL-3.0](https://www.ultralytics.com/legal/agpl-3-0-software-license) and a
paid [Enterprise License](https://www.ultralytics.com/license). Under AGPL-3.0,
if you distribute this project or a service built on it (including over a
network), you're generally required to open-source the full solution under
AGPL-3.0 — application code, training/inference scripts, and trained model
weights included. An Enterprise License removes this open-source requirement
for closed-source or commercial use. See
[Ultralytics licensing](https://www.ultralytics.com/license) for details.

## Requirements

See [Prerequisites](#prerequisites) above for accounts, quotas, and hardware. For
the full Python dependency list, see `pyproject.toml` (local/dev) and
`scripts/pyproject.toml` (the pinned CUDA build used on GPU instances).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

