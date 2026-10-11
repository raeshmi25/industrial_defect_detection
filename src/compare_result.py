
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODEL_PATH = PROJECT_ROOT / "models" / "bottle_baseline.npz"
RESULTS_DIR = PROJECT_ROOT / "results"

IMAGE_SIZE = (128, 128)


def main():
    # Path to the defective test image
    test_image_path = (
        DATASETS_DIR
        / "bottle"
        / "test"
        / "broken_large"
        / "000.png"
    )

    # IMPORTANT: Ground-truth filenames end with _mask.png
    mask_path = (
        DATASETS_DIR
        / "bottle"
        / "ground_truth"
        / "broken_large"
        / "000_mask.png"
    )

    # Check that the required files exist
    if not MODEL_PATH.exists():
        print("Baseline model not found.")
        print("Run: python src/train_baseline.py")
        return

    if not test_image_path.exists():
        print(f"Test image not found: {test_image_path}")
        return

    if not mask_path.exists():
        print(f"Ground-truth mask not found: {mask_path}")
        return

    # Load the trained baseline model
    data = np.load(MODEL_PATH)
    mean_image = data["mean_image"]
    std_image = data["std_image"]

    # Load the original defective image
    original_bgr = cv2.imread(str(test_image_path))

    if original_bgr is None:
        print("Could not read the test image.")
        return

    original_rgb = cv2.cvtColor(
        original_bgr, cv2.COLOR_BGR2RGB
    )

    # Convert test image to grayscale and resize it
    gray = cv2.imread(
        str(test_image_path), cv2.IMREAD_GRAYSCALE
    )
    gray = cv2.resize(gray, IMAGE_SIZE)
    image = gray.astype(np.float32) / 255.0

    # Calculate pixel-wise anomaly scores
    scores = np.abs(image - mean_image) / (
        std_image + 0.05
    )

    # Scale scores for visualization
    scores = np.clip(scores, 0, 10) / 10.0
    heatmap = (scores * 255).astype(np.uint8)

    # Load the ground-truth defect mask
    mask = cv2.imread(
        str(mask_path), cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        print("Could not read the ground-truth mask.")
        return

    mask = cv2.resize(
        mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    # Prepare the original image at the heatmap resolution
    resized_original = cv2.resize(
        original_rgb, IMAGE_SIZE
    )

    # Create a four-panel comparison
    fig, axes = plt.subplots(
        1, 4, figsize=(16, 5)
    )

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Defective Image")

    axes[1].imshow(
        mask, cmap="gray", vmin=0, vmax=255
    )
    axes[1].set_title("Ground-Truth Mask")

    axes[2].imshow(
        heatmap, cmap="inferno", vmin=0, vmax=255
    )
    axes[2].set_title("Baseline Anomaly Heatmap")

    axes[3].imshow(resized_original)
    axes[3].imshow(
        heatmap, cmap="inferno", alpha=0.5,
        vmin=0, vmax=255
    )
    axes[3].set_title("Heatmap Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    # Save the comparison figure
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "comparison_000.png"

    plt.savefig(
        output_path, dpi=150, bbox_inches="tight"
    )

    print(f"Test image: {test_image_path}")
    print(f"Ground-truth mask: {mask_path}")
    print(f"Comparison saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
