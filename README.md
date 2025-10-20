# ds-281-facial-recognition

This repository contains materials for a facial emotion recognition project built around the FER-2013 dataset. The work includes dataset folders, Jupyter notebooks for exploration and modeling, and a trained PyTorch model checkpoint used in the notebooks.

## Summary

Goal: build and evaluate models that classify facial expressions into emotion categories (angry, disgust, fear, happy, neutral, sad, surprise) using the FER-2013 dataset.

This repo was used for final project work in DS 281 and contains data folders, Jupyter notebooks for exploration and modeling, and at least one trained PyTorch model checkpoint.

## Repository structure

- `fer2013/` - the dataset split into `train/` and `test/` with subfolders for each emotion class (angry, disgust, fear, happy, neutral, sad, surprise). Use these folders for training scripts or creating PyTorch datasets from folders.
- `Notebooks/` - Jupyter notebooks used for EDA and modeling. Notable files:
  - `Marcelino_project_eda.ipynb` - exploratory data analysis.
  - `Xinyi_simple_feature_1.ipynb` and `Xinyi_complex_feature_2.ipynb` - feature engineering and modeling experiments.
  - `best_efficientnetb0_fer2013.pth` - a trained EfficientNet-B0 PyTorch checkpoint used in experiments.
- `Final Project Proposal/` and `Final Project Rubric/` - project planning and grading artifacts.

## Dataset (FER-2013)

FER-2013 is a widely used facial expression recognition dataset with grayscale face crops labeled for 7 emotions. The dataset copies in this repository are organized into class subfolders under `fer2013/train/` and `fer2013/test/` which makes them compatible with torchvision's `ImageFolder` or other folder-based dataset loaders.

If you do not have the raw FER-2013 CSV or images, obtain a copy from the original source and arrange the images into the same directory structure used here.

## Quick start

Prerequisites

- macOS or Linux (Windows supported with minor path differences)
- Python 3.8+ (3.9/3.10 recommended)
- pip and virtualenv or venv

Create and activate a virtual environment and install typical dependencies. If a `requirements.txt` is present, prefer that; otherwise install the common packages used in the notebooks:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# If requirements.txt exists:
pip install -r requirements.txt || true
# Common packages used in notebooks:
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn jupyterlab pillow tqdm
```

Open the notebooks

```bash
jupyter lab  # or `jupyter notebook`
```

Then open any notebook in `Notebooks/` to explore EDA and model experiments.

## Loading the provided model checkpoint (PyTorch)

The file `Notebooks/best_efficientnetb0_fer2013.pth` is a PyTorch checkpoint saved with `torch.save(...)`. Example to load a model in a notebook or script:

```python
import torch
from torchvision import models

# Example: create an EfficientNet-B0 backbone and load state dict (adjust to how the model was saved)
model = models.efficientnet_b0(pretrained=False)
# If the checkpoint contains a full state dict: state = torch.load('Notebooks/best_efficientnetb0_fer2013.pth')
# If it contains only model weights: model.load_state_dict(state)
state = torch.load('Notebooks/best_efficientnetb0_fer2013.pth', map_location='cpu')
if isinstance(state, dict) and 'model_state_dict' in state:
	model.load_state_dict(state['model_state_dict'])
else:
	try:
		model.load_state_dict(state)
	except Exception:
		# If the checkpoint contains a wrapped or different format, inspect keys:
		print(type(state), state.keys() if hasattr(state, 'keys') else None)

model.eval()
```

Adjust the loader code to match how the checkpoint was saved in the notebooks (full checkpoint vs. state_dict-only).

## Reproducing experiments

1. Ensure images are arranged under `fer2013/train/<class>` and `fer2013/test/<class>`.
2. Open `Notebooks/Xinyi_simple_feature_1.ipynb` or `Xinyi_complex_feature_2.ipynb` and follow the preprocessing and training cells. The notebooks contain the data transforms and training loops used in experiments.
3. For longer training runs, use a machine with a CUDA GPU and update device logic in the notebook or export the training cells into a script.

## Notes and caveats

- The repository currently stores processed image folders rather than a single FER-2013 CSV. Confirm that the file structure matches your training code expectations.
- Checkpoints may have been saved with different model wrappers (e.g., `nn.DataParallel`) — if you encounter key mismatches when loading, remove unwanted prefixes like `'module.'` from the state dict keys.

## License & Citation

This repo contains student work for an academic course. Please contact the repository owner for reuse permissions. If you use FER-2013 or any models from this work, cite the original FER-2013 dataset and the appropriate course/project authors.