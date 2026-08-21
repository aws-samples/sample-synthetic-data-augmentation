# YOLO Evaluation Pipeline

Evaluate and compare YOLO models trained with different data strategies
(baseline vs. synthetic-augmented) on a held-out, real-only test set.

These scripts assume `SDA_S3_BUCKET` is set and expect two trained models —
a baseline and a synthetic-augmented model — but any set of models can be
passed explicitly (see `extract_per_class.py --models`).

## Setup

### 1. Download models from S3

SageMaker training outputs are tarballed. Download each `model.tar.gz`, extract,
and rename to something meaningful:

```bash
mkdir -p models && cd models

# Replace <job-name> with your SageMaker training job names
aws s3 cp s3://$SDA_S3_BUCKET/training_runs/<baseline-job>/output/model.tar.gz ./baseline.tar.gz
aws s3 cp s3://$SDA_S3_BUCKET/training_runs/<synthetic-job>/output/model.tar.gz ./synthetic.tar.gz

mkdir -p baseline synthetic
tar -xzf baseline.tar.gz -C baseline
tar -xzf synthetic.tar.gz -C synthetic

cp baseline/best.pt ./best_baseline.pt
cp synthetic/best.pt ./best_synthetic.pt
cd ..
```

### 2. Download the test dataset (run once)

```bash
export SDA_S3_BUCKET=amzn-s3-demo-bucket
python download_testdata.py
```

Downloads images/labels from S3 and creates `test_data/val_dataset.yaml`.

## Evaluation

### Per-class mAP breakdown

The key analysis — does synthetic data improve **person** detection specifically:

```bash
python extract_per_class.py \
    --data-yaml ./test_data/val_dataset.yaml \
    --models-dir ./models

# Or pass models explicitly:
python extract_per_class.py \
    --data-yaml ./test_data/val_dataset.yaml \
    --models "baseline:./models/best_baseline.pt,synthetic:./models/best_synthetic.pt"
```

### Aggregate metrics with plots

```bash
python run_eval.py --model-path ./models/best_baseline.pt --data-yaml ./test_data/val_dataset.yaml
python run_eval.py --model-path ./models/best_synthetic.pt --data-yaml ./test_data/val_dataset.yaml
```

Outputs go to `best_baseline_output/`, `best_synthetic_output/`, etc.

### Compare results

```bash
# Side-by-side comparisons (confusion matrices, PR curves, predictions)
python compare_results.py \
    --baseline-dir ./best_baseline_output \
    --synthetic-dir ./best_synthetic_output \
    --compare all

# Options: all, confusion, curves, predictions, labels
```

### Find missed detections

Identify images where the model failed to detect objects:

```bash
python find_missed_detections.py --model-path ./models/best_baseline.pt --class-name person --output-dir ./missed_persons_baseline
python find_missed_detections.py --model-path ./models/best_synthetic.pt --class-name person --output-dir ./missed_persons_synthetic
```

Output images show: GREEN = ground truth, BLUE = target-class predictions (`--class-name`; not correctness-checked against ground truth), RED = other-class predictions.

## Training (for reference)

```bash
export SDA_S3_BUCKET=amzn-s3-demo-bucket
export AWS_DEFAULT_REGION=us-west-2

python ../yolo_training/launch_training.py \
    --s3-data s3://$SDA_S3_BUCKET/datasets/openimages_subset/ \
    --s3-output s3://$SDA_S3_BUCKET/training_runs \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/YourSageMakerRole \
    --dataset-yaml data.yaml  # or data-with-synthetic.yaml
```
