<<<<<<< HEAD

# Industrial Defect Detection and Localization

A Computer Vision project to detect abnormal patterns in industrial
products and localize potential defects using anomaly heatmaps.

## Objectives

- Inspect industrial product images.
- Identify normal and anomalous samples.
- Localize potential defects.
- Evaluate predictions against ground-truth masks.
- Build a reproducible, documented Computer Vision pipeline.

## Dataset

**MVTec Anomaly Detection (MVTec AD)**

The dataset will be downloaded separately and is not included in
this repository.

Initial experiments will focus on the bottle category.

## Technologies

- Python 3.12
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Scikit-learn

## Project Structure

```text
industrial-defect-detection/
├── datasets/
├── models/
├── notebooks/
├── results/
├── src/
│   ├── inspect_dataset.py
│   └── demo_baseline.py
├── .gitignore
└── README.md
```

## Current Progress

- [x] Python virtual environment configured.
- [x] Initial dataset inspection script created.
- [x] Synthetic image-processing demonstration.
- [ ] MVTec AD dataset downloaded and verified.
- [ ] Anomaly detection model implemented.
- [ ] Defect localization evaluated.
- [ ] Results documented.

## Run the Dataset Inspector

Activate the virtual environment, then run:

```powershell
python src/inspect_dataset.py
```

## Disclaimer

The synthetic heatmap demo is a basic image-processing experiment.
It is not a trained anomaly-detection model.

## License

The dataset is subject to its own license terms.
Check the applicable dataset license before using or redistributing it.
=======
# industrial_defect_detection
>>>>>>> dcf8e3ac622b23f3edd017d8ba572cf00ea94e34
