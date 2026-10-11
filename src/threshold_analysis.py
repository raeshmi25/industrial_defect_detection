
from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "bottle"
MODEL_PATH = PROJECT_ROOT / "models" / "bottle_baseline.npz"
RESULTS_DIR = PROJECT_ROOT / "results"

IMAGE_SIZE = (128, 128)

# Explore several thresholds to understand the trade-off.
THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, IMAGE_SIZE)
    return image.astype(np.float32) / 255.0


def main():
    if not MODEL_PATH.exists():
        print("Model missing. Run python src/train_baseline.py")
        return

    model = np.load(MODEL_PATH)
    mean_image = model["mean_image"]
    std_image = model["std_image"]

    test_dir = DATASET_DIR / "test"
    mask_root = DATASET_DIR / "ground_truth"

    if not test_dir.is_dir() or not mask_root.is_dir():
        print("Test or ground-truth directory not found.")
        return

    # Store all continuous scores and labels across test images.
    score_parts = []
    label_parts = []
    image_count = 0

    for category_dir in sorted(test_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name

        for image_path in sorted(category_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in {
                ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
            }:
                continue

            image = load_gray(image_path)
            if image is None:
                continue

            if category == "good":
                mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
            else:
                mask_path = (
                    mask_root / category
                    / f"{image_path.stem}_mask.png"
                )
                if not mask_path.exists():
                    print(f"Missing mask, skipping: {mask_path}")
                    continue

                mask = cv2.imread(
                    str(mask_path), cv2.IMREAD_GRAYSCALE
                )
                if mask is None:
                    continue

                mask = cv2.resize(
                    mask, IMAGE_SIZE,
                    interpolation=cv2.INTER_NEAREST
                )

            scores = np.abs(image - mean_image) / (std_image + 0.05)

            score_parts.append(scores.ravel())
            label_parts.append((mask > 0).astype(np.uint8).ravel())
            image_count += 1

    if not score_parts:
        print("No images were available for analysis.")
        return

    scores = np.concatenate(score_parts)
    labels = np.concatenate(label_parts).astype(bool)

    rows = []

    print(f"\nImages analyzed: {image_count}")
    print("\nThreshold | Precision | Recall | IoU")
    print("-" * 43)

    for threshold in THRESHOLDS:
        predicted = scores >= threshold

        tp = int(np.sum(predicted & labels))
        fp = int(np.sum(predicted & ~labels))
        fn = int(np.sum(~predicted & labels))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0

        rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "iou": iou,
        })

        print(
            f"{threshold:9.2f} | {precision:9.4f} | "
            f"{recall:6.4f} | {iou:.4f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "threshold_analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Save a plot of the metric trade-offs.
    thresholds = [row["threshold"] for row in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(
        thresholds, [row["precision"] for row in rows],
        marker="o", label="Precision"
    )
    plt.plot(
        thresholds, [row["recall"] for row in rows],
        marker="o", label="Recall"
    )
    plt.plot(
        thresholds, [row["iou"] for row in rows],
        marker="o", label="IoU"
    )
    plt.xlabel("Anomaly score threshold")
    plt.ylabel("Metric value")
    plt.title("Bottle Baseline: Threshold Analysis")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = RESULTS_DIR / "threshold_analysis.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()

    print(f"\nCSV saved: {csv_path}")
    print(f"Chart saved: {plot_path}")
    print(
        "\nExploratory analysis only: these thresholds use test-set "
        "labels for comparison. Do not use this sweep to claim "
        "unbiased final test performance."
    )


if __name__ == "__main__":
    main()
