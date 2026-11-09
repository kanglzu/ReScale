# Data preparation

[← ReScale](../README.md)

Prepare your image/mask pairs using the layouts below, then run `prepare_data.py`
with the dataset name and folder path. The script records local paths and checks
the supplied splits; it does not resize images or create new folds.

| Dataset (`--dataset`) | Images | Expected paths below dataset root | Target/preparation |
|---|---:|---|---|
| CVC-ClinicDB | 612 | `Original/*.png`, `Ground Truth/*.png` | Polyp masks |
| ISIC2018 | 2,594 | `ISIC2018_Input/*.jpg`, `ISIC2018_GroundTruth/*_segmentation.png` | Task 1 lesion masks |
| BUSI | 647 | `images/benign_001.png`, `masks/benign_001.png`, etc. | Benign and malignant images; union of all masks for each image; normal cases excluded |
| PAPILA | 488 | `images/RET002OD.jpg`, `masks/RET002OD.png`, etc. | First-expert optic-disc contours rasterized on the native grid; original JPEGs preserved |
| PTX498 | 498 | `images/SiteA_0001.png`, `masks/SiteA_0001.png`, etc. | v2-fix source, image resized to 256² with OpenCV linear interpolation, binary mask with nearest interpolation |
| FUSeg2021 | 1,010 | `images/0001.png`, `masks/0001.png`, etc. | Labeled training/validation release pooled; image/mask resized to 256² with linear/nearest interpolation |

The loader uses three input channels, repeating grayscale values when needed.
Mask values greater than zero are treated as foreground. Each file in `manifests/`
lists the expected image/mask pairs and dataset-specific preprocessing details.

Use the existing files in `splits/` for the five folds. PAPILA splits group images
by patient; grouping details for each dataset are recorded in its split file.

Images and masks are not included in the repository.
