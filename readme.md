# Machine Learning Prediction of Live Birth After IVF Using Adenomyosis Features

This repository contains the code for the paper: **"Machine learning prediction of live birth after IVF using the Morphological Uterus Sonographic Assessment group features of adenomyosis"**

## Overview

This project develops XGBoost-based machine learning models to predict live birth rates (LBR) following in vitro fertilization (IVF) treatment. The models incorporate ultrasound features of adenomyosis based on the Morphological Uterus Sonographic Assessment (MUSA) criteria.

Two model variants are provided:
- **Original model**: Core clinical and ultrasound features
- **Extended model**: Original features plus embryo transfer variables (Cleavage_stage, Blastocyst_stage, FET) added per reviewer request.

## Repository Structure

```
ivf_lbr_prediction/
├── data/                          # Data directory (data files not included)
│   └── kfold_ids.json
├── data_process/                 # Data preprocessing utilities
│   ├── datainfo.py               # Variable definitions and metadata
│   ├── preprocess_data.py        # Data cleaning and preparation
│   └── create_kfold_splits.py    # K-fold cross-validation split creation
├── modeling/                      # Model training and evaluation
│   ├── model_config.py           # Configuration and paths
│   ├── data_loader.py            # Data loading utilities
│   ├── optuna_helper.py          # Hyperparameter optimization functions
│   ├── train_model.py            # Model training script
│   └── evaluate_model.py         # Model evaluation script
├── results/                       # Model outputs (generated during execution)
│   ├── original_model/
│   │   ├── studies/              # Optuna optimization studies
│   │   ├── feature_importance.xlsx
│   │   └── roc_curve.png
│   └── extended_model/
│       ├── studies/
│       ├── feature_importance.xlsx
│       └── roc_curve.png
└── requirements.txt               # Python package dependencies
```

## Requirements

- Python 3.11
- See `requirements.txt` for complete package dependencies

### Installation

**Note**: This repository contains code only. Patient data is not included due to privacy regulations. You will need your own dataset with similar structure.

1. Clone the repository:
```bash
git clone https://github.com/F-Ursus/ivf_lbr_prediction.git
cd ivf_lbr_prediction
```

2. Create a conda environment (recommended):
```bash
conda create -n ivf_lbr python=3.11
conda activate ivf_lbr
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

First, preprocess the raw data:

```bash
python -m data_process.preprocess_data
```

This creates `LBR_train.csv` and `LBR_test.csv` in the `data/` folder.

**Note**: The preprocessing script expects a file named `XGBoost_Adeno_ART_LBR_250823.xlsx` in the `data/` folder.

### 2. Create K-Fold Cross-Validation Splits

Create stratified k-fold cross-validation splits (first time only):

```bash
python -m data_process.create_kfold_splits
```

This creates `kfold_ids.json` which ensures identical CV folds across all experiments. The script will:
- Create new splits if `kfold_ids.json` doesn't exist
- Verify existing splits and offer options to keep, verify reproducibility, or recreate

**Important**: Only run this once! The same splits are reused for all models to ensure fair comparison. Recreating splits will change cross-validation results.

### 3. Training a New Model

To train a new model with hyperparameter optimization:

```bash
# Train original model (50 trials)
python -m modeling.train_model --variant original --mode new --trials 50

# Train extended model (50 trials)
python -m modeling.train_model --variant extended --mode new --trials 50
```

### 4. Continuing an Existing Study

To continue hyperparameter optimization from a saved study:

```bash
# Continue extended model optimization (100 trials per iteration, 2 iterations)
python -m modeling.train_model --variant extended --mode continue --trials 100 --iterations 2
```

### 5. Evaluating a Trained Model

To evaluate the best model from a study:

```bash
# Evaluate extended model (save results and show plots)
python -m modeling.evaluate_model --variant extended --save

# Evaluate without displaying plots (useful for batch processing)
python -m modeling.evaluate_model --variant extended --save --no-show
```

### Viewing Configuration

To print all configuration paths and parameters:

```bash
python -m modeling.model_config
```

## Model Variants

### Original Model
Core clinical and ultrasound features, excluding:
- `Cleavage_stage`
- `Blastocyst_stage`
- `FET` (Frozen Embryo Transfer)

### Extended Model
Includes all features from the original model plus the three embryo transfer variables listed above. Added in response to reviewer request.

## Output Files

After training and evaluation, the following files are generated:

- **`results/<variant>/studies/`**: Saved Optuna studies (pickle files)
- **`results/<variant>/feature_importance.xlsx`**: SHAP-based feature importance rankings
- **`results/<variant>/roc_curve.png`**: ROC curves for training and test sets

## Key Features

- **Hyperparameter optimization**: Uses Optuna with Tree-structured Parzen Estimator (TPE) sampling
- **Cross-validation**: Stratified k-fold cross-validation (5 folds) with saved splits
- **Feature importance**: SHAP values for model interpretability
- **Model evaluation**: ROC-AUC, accuracy at multiple thresholds

## Reproducibility

To ensure reproducibility:
1. All package versions are pinned in `requirements.txt`
2. Random seeds are fixed (RANDOM_STATE = 13)
3. **K-fold splits** are pre-defined and saved in `data/kfold_ids.json`:
   - Created using `StratifiedKFold` with `n_splits=5`, `shuffle=True`, `random_state=13`
   - Ensures identical cross-validation folds across all experiments
   - Patient IDs are saved (not actual data) for transparency
4. Exact Python version (3.11) is specified

## Data

**Note**: The datasets are not included in this repository due to privacy considerations. 

### Required Data Files

The code expects the following files in the `data/` directory:

**For preprocessing (Step 1):**
- `XGBoost_Adeno_ART_LBR_250823.xlsx`: Raw data file (not included)

**Generated by preprocessing:**
- `LBR_train.csv`: Training data with patient features and outcomes
- `LBR_test.csv`: Test data with patient features and outcomes

**Generated by k-fold split creation (Step 2):**
- `kfold_ids.json`: Pre-defined k-fold cross-validation splits (patient IDs only)


## Citation
This code accompanies the paper by Alson, S., Björnsson, O., et al.

If you use this code, please cite:

```
[Citation will be added upon publication]
```

**Software archive:**

```
Björnsson, O., & Alson, S. (2024). ivf_lbr_prediction: Machine Learning Prediction 
of Live Birth After IVF Using Adenomyosis Features (v1.0.0). Zenodo. 
https://doi.org/10.5281/zenodo.XXXXXXX
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
