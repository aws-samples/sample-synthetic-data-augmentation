"""Side-by-side visualization comparison of YOLO evaluation outputs (2-way or 3-way)."""
from pathlib import Path

import click
import matplotlib.pyplot as plt
from PIL import Image


COMPARISONS = {
    "confusion": ["confusion_matrix.png", "confusion_matrix_normalized.png"],
    "curves": ["BoxF1_curve.png", "BoxP_curve.png", "BoxPR_curve.png", "BoxR_curve.png"],
    "predictions": ["val_batch0_pred.jpg", "val_batch1_pred.jpg", "val_batch2_pred.jpg"],
    "labels": ["val_batch0_labels.jpg", "val_batch1_labels.jpg", "val_batch2_labels.jpg"],
}


def compare_images(
    dirs: list[tuple[str, Path]],
    image_names: list[str],
    output_path: Path | None = None,
) -> None:
    """Create side-by-side comparison of images from N directories."""
    n_cols = len(dirs)
    valid_images = [
        name for name in image_names
        if all((d / name).exists() for _, d in dirs)
    ]
    if not valid_images:
        print("No matching images found")
        return

    n_rows = len(valid_images)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[a] for a in axes]

    for row, name in enumerate(valid_images):
        for col, (label, d) in enumerate(dirs):
            img = Image.open(d / name)
            axes[row][col].imshow(img)
            axes[row][col].set_title(f"{label}: {name}", fontsize=10)
            axes[row][col].axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


@click.command()
@click.option("--baseline-dir", default="./best_baseline_output", help="Baseline output dir")
@click.option("--synthetic-dir", default="./best_synthetic_output", help="Synthetic output dir")
@click.option("--extra-dir", default=None, help="Optional third output dir for a 3-way comparison")
@click.option("--extra-label", default="Extra", help="Legend label for --extra-dir")
@click.option("--compare", type=click.Choice(["all", "confusion", "curves", "predictions", "labels"]), default="all")
@click.option("--output-dir", default=None, help="Save dir (interactive if not set)")
def main(baseline_dir: str, synthetic_dir: str, extra_dir: str | None, extra_label: str, compare: str, output_dir: str | None) -> None:
    """Compare YOLO evaluation outputs side-by-side (2-way or 3-way)."""
    dirs: list[tuple[str, Path]] = [("Baseline", Path(baseline_dir)), ("Synthetic", Path(synthetic_dir))]
    if extra_dir:
        dirs.append((extra_label, Path(extra_dir)))

    for label, d in dirs:
        if not d.exists():
            raise click.ClickException(f"{label} dir not found: {d}")

    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    comparisons = list(COMPARISONS.keys()) if compare == "all" else [compare]
    for comp_type in comparisons:
        print(f"\nComparing: {comp_type}")
        out_file = output_path / f"comparison_{comp_type}.png" if output_path else None
        compare_images(dirs, COMPARISONS[comp_type], out_file)


if __name__ == "__main__":
    main()
