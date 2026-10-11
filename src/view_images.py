
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def main():
    images = [
        p for p in DATASETS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        print("No dataset images found yet.")
        print("Once the dataset is extracted into datasets/, run this again.")
        return

    print(f"Found {len(images)} images.")
    selected = images[:6]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, path in zip(axes.flat, selected):
        image = cv2.imread(str(path))
        if image is None:
            ax.set_title("Could not load image")
            ax.axis("off")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_title(path.parent.name)
        ax.axis("off")

    for ax in axes.flat[len(selected):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
