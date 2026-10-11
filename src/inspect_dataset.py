
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

def main():
    print("=" * 55)
    print("INDUSTRIAL DEFECT DETECTION - DATASET INSPECTOR")
    print("=" * 55)
    print(f"Project folder: {PROJECT_ROOT}")
    print(f"Dataset folder: {DATASETS_DIR}")

    if not DATASETS_DIR.exists():
        print("ERROR: Dataset folder does not exist.")
        return

    # Show the first-level folders without scanning every image.
    print("\nFolders found inside datasets:")
    for item in DATASETS_DIR.iterdir():
        if item.is_dir():
            print(f"  [Folder] {item.name}")
        elif item.is_file():
            print(f"  [File]   {item.name}")

    # Look for the bottle category.
    bottle_folders = [
        p for p in DATASETS_DIR.iterdir()
        if p.is_dir() and p.name.lower() == "bottle"
    ]

    # Check one extra folder level if needed.
    if not bottle_folders:
        for parent in DATASETS_DIR.iterdir():
            if parent.is_dir():
                candidate = parent / "bottle"
                if candidate.is_dir():
                    bottle_folders.append(candidate)

    if bottle_folders:
        bottle = bottle_folders[0]
        print(f"\nBottle category found: {bottle}")

        for name in ("train", "test", "ground_truth"):
            folder = bottle / name
            print(f"  {name}/: {'Found' if folder.is_dir() else 'Not found'}")
    else:
        print("\nBottle category not found at the expected levels.")
        print("We will check the folder structure together.")

if __name__ == "__main__":
    main()
