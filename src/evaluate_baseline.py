
from pathlib import Path
import csv

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "bottle"
MODEL_PATH = PROJECT_ROOT / "models" / "bottle_baseline.npz"
RESULTS_DIR = PROJECT_ROOT / "results"

IMAGE_SIZE = (128, 128)

# Fixed threshold for a first, transparent baseline evaluation.
# We will not tune this threshold on the test masks.
ANOMALY_THRESHOLD = 3.0


def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, IMAGE_SIZE)
    return image.astype(np.float32) / 255.0


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run: python src/train_baseline.py")
        return

    test_dir = DATASET_DIR / "test"
    mask_root = DATASET_DIR / "ground_truth"

    if not test_dir.is_dir() or not mask_root.is_dir():
        print("Bottle test or ground_truth folder was not found.")
        return

    model = np.load(MODEL_PATH)
    mean_image = model["mean_image"]
    std_image = model["std_image"]

    all_scores = []
    all_labels = []
    rows = []
    skipped = 0

    # Evaluate good images and defective images across all test categories.
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
                skipped += 1
                continue

            # Continuous anomaly scores, before heatmap color conversion.
            scores = np.abs(image - mean_image) / (std_image + 0.05)

            if category == "good":
                # Normal test images have no annotated defect pixels.
                mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
            else:
                mask_path = (
                    mask_root / category / f"{image_path.stem}_mask.png"
                )

                if not mask_path.exists():
                    print(f"Skipping missing mask: {mask_path}")
                    skipped += 1
                    continue

                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    skipped += 1
                    continue

                mask = cv2.resize(
                    mask,
                    IMAGE_SIZE,
                    interpolation=cv2.INTER_NEAREST
                )

            labels = (mask > 0).astype(np.uint8).ravel()
            flat_scores = scores.ravel()

            # Keep the pixel scores and ground-truth labels for pooled metrics.
            all_scores.append(flat_scores)
            all_labels.append(labels)

            predicted = flat_scores >= ANOMALY_THRESHOLD
            actual = labels == 1

            tp = int(np.sum(predicted & actual))
            fp = int(np.sum(predicted & ~actual))
            fn = int(np.sum(~predicted & actual))

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0

            rows.append({
                "image": str(image_path.relative_to(DATASET_DIR)),
                "category": category,
                "precision_at_threshold": precision,
                "recall_at_threshold": recall,
                "iou_at_threshold": iou,
                "threshold": ANOMALY_THRESHOLD,
            })

    if not rows:
        print("No images could be evaluated. Check the dataset structure.")
        return

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    predicted = scores >= ANOMALY_THRESHOLD
    actual = labels == 1

    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0

    if len(np.unique(labels)) == 2:
        auc = roc_auc_score(labels, scores)
    else:
        auc = float("nan")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "baseline_evaluation_per_image.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== BOTTLE BASELINE EVALUATION ==========")
    print(f"Images evaluated: {len(rows)}")
    print(f"Images skipped: {skipped}")
    print(f"Anomaly threshold: {ANOMALY_THRESHOLD:.2f}")
    print(f"True positives (pixels): {tp}")
    print(f"False positives (pixels): {fp}")
    print(f"False negatives (pixels): {fn}")
    print(f"True negatives (pixels): {tn}")
    print(f"Pixel precision: {precision:.4f}")
    print(f"Pixel recall:    {recall:.4f}")
    print(f"Pixel IoU:       {iou:.4f}")
    print(f"Pixel ROC-AUC:   {auc:.4f}")
    print("\nPer-image metrics saved to:")
    print(csv_path)
    print("\nNote: these are pixel-level metrics pooled across evaluated")
    print("images. The fixed threshold is a baseline, not an optimized one.")


if __name__ == "__main__":
    main()
