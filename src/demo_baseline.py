
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Create a synthetic industrial surface for testing only.
    rng = np.random.default_rng(42)
    image = np.full((256, 256, 3), 180, dtype=np.uint8)
    noise = rng.normal(0, 3, image.shape)
    image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

    # Simulate a dark scratch.
    cv2.line(image, (65, 75), (195, 155), (65, 65, 65), 4)

    # Simple baseline: find pixels darker than the expected surface.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    heatmap = np.clip((180 - gray).astype(float), 0, 255)

    # Save a demonstration image.
    output_path = RESULTS_DIR / "synthetic_baseline.png"

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Synthetic surface with scratch")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap="inferno", vmin=0, vmax=120)
    plt.title("Simple anomaly-like heatmap")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"Demo saved to: {output_path}")

if __name__ == "__main__":
    main()
