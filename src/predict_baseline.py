
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "bottle_baseline.npz"
RESULTS_DIR = PROJECT_ROOT / "results" / "heatmaps"

def main():
    if not MODEL_PATH.exists():
        print("Baseline model not found.")
        print("First, extract the dataset and run:")
        print("python src/train_baseline.py")
        return

    model = np.load(MODEL_PATH)
    mean_image = model["mean_image"]
    std_image = model["std_image"]

    category = None
    datasets_dir = PROJECT_ROOT / "datasets"

    for path in datasets_dir.rglob("bottle"):
        if (path / "test").is_dir():
            category = path
            break

    if category is None:
        print("Bottle test images not found in datasets/.")
        return

    # Select one test image to demonstrate the pipeline.
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    test_images = sorted(
        p for p in (category / "test").rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not test_images:
        print("No test images found.")
        return

    image_path = test_images[0]
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Could not read image: {image_path}")
        return

    image = cv2.resize(image, (128, 128)).astype(np.float32) / 255.0

    # Simple pixel deviation baseline.
    score = np.abs(image - mean_image) / (std_image + 0.05)
    heatmap = np.clip(score / 5.0 * 255.0, 0, 255).astype(np.uint8)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{image_path.stem}_heatmap.png"

    color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(output_path), color_heatmap)

    print(f"Test image: {image_path}")
    print(f"Heatmap saved to: {output_path}")
    print("Note: this is a simple baseline, not a validated defect prediction.")

if __name__ == "__main__":
    main()
