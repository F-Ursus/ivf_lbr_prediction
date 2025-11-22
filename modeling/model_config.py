"""
Configuration for model training and evaluation.
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    root_str = str(project_root)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from data_process import datainfo

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
TRAIN_FILE = DATA_FOLDER / "LBR_train.csv"
TEST_FILE = DATA_FOLDER / "LBR_test.csv"
KFOLD_JSON_FILE = DATA_FOLDER / "kfold_ids.json"

# Model variants use different output directories
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIRS = {
    "original": RESULTS_DIR / "original_model",
    "extended": RESULTS_DIR / "extended_model",
}
STUDY_DIRS = {
    "original": RESULTS_DIR / "original_model" / "studies",
    "extended": RESULTS_DIR / "extended_model" / "studies",
}

# Extended model variables (additional clinical variables requested by reviewers)
# These are excluded from the 'original' model but included in the 'extended' model
EXTENDED_VARS = datainfo.extended_variables

# Training configuration
N_JOBS = 20
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 15
MODEL_EVAL_METRIC = "avg_auc"
RANDOM_STATE = 13


def get_objective_kwargs() -> dict[str, int | str]:
    """Return standard objective function kwargs for optuna."""
    return {
        "n_splits": N_SPLITS,
        "n_jobs": N_JOBS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "random_state": RANDOM_STATE,
        "model_eval_metric": MODEL_EVAL_METRIC,
    }


def ensure_directories() -> None:
    """Create necessary output directories if they don't exist."""
    for model_variant in ["original", "extended"]:
        MODEL_DIRS[model_variant].mkdir(parents=True, exist_ok=True)
        STUDY_DIRS[model_variant].mkdir(parents=True, exist_ok=True)


def print_paths() -> None:
    """Print all important paths for debugging."""
    print("=== Configuration Paths ===")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Folder: {DATA_FOLDER}")
    print(f"Train File: {TRAIN_FILE}")
    print(f"Test File: {TEST_FILE}")
    print(f"K-Fold JSON File: {KFOLD_JSON_FILE}")
    print("\n=== Results Directories ===")
    print(f"Results Directory: {RESULTS_DIR}")
    for variant, path in MODEL_DIRS.items():
        print(f"Model Directory ({variant}): {path}")
    for variant, path in STUDY_DIRS.items():
        print(f"Study Directory ({variant}): {path}")
    print("\n=== Training Configuration ===")
    print(f"Number of Jobs: {N_JOBS}")
    print(f"Number of Splits: {N_SPLITS}")
    print(f"Early Stopping Rounds: {EARLY_STOPPING_ROUNDS}")
    print(f"Model Evaluation Metric: {MODEL_EVAL_METRIC}")
    print(f"Random State: {RANDOM_STATE}")


if __name__ == "__main__":
    ensure_directories()
    print_paths()
