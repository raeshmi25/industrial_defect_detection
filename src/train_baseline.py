
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGE_SIZE = (128, 128)


def find_bottle_category():
    """Find the bottle folder, even if the archive adds an extra parent folder."""
    for path in DATASETS_DIR.rglob("bottle"):
        if path.is_dir() and (path / "train" / "good").is_dir():
            return path
    return None


def load_images(folder):
    images = []

    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        image = cv2.resize(image, IMAGE_SIZE)
        images.append(image.astype(np.float32) / 255.0)

    return images


def main():
    category = find_bottle_category()

    if category is None:
        print("Bottle training folder not found yet.")
        print("Expected structure: bottle/train/good/")
        print("Extract the dataset under the project's datasets folder.")
        return

    normal_folder = category / "train" / "good"
    normal_images = load_images(normal_folder)

    if len(normal_images) < 2:
        print("Not enough normal training images were found.")
        return

    # Learn the average pixel appearance of normal images.
    stack = np.stack(normal_images)
    mean_image = stack.mean(axis=0)
    std_image = stack.std(axis=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "bottle_baseline.npz"

    np.savez(
        model_path,
        mean_image=mean_image,
        std_image=std_image,
        image_size=np.array(IMAGE_SIZE),
    )

    print(f"Training images used: {len(normal_images)}")
    print(f"Baseline model saved to: {model_path}")
    print("Next, we will generate and evaluate anomaly heatmaps.")


if __name__ == "__main__":
    main()
