"""
Shared data loading and preprocessing utilities.
"""

import json
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from data_process import datainfo


def remove_variables(
    df: pd.DataFrame, to_remove: Optional[list[str]] = None
) -> pd.DataFrame:
    """Remove specified columns from dataframe along with mandatory removals."""
    mandatory_removals = datainfo.actual_removals
    combined_removals = set(mandatory_removals + (to_remove or []))
    existing_removals = combined_removals.intersection(df.columns)
    return df.drop(columns=list(existing_removals))


def set_cat_variables(df: pd.DataFrame, extended_vars: list[str]) -> pd.DataFrame:
    """Convert categorical and boolean columns to 'category' dtype."""
    cat_vars = datainfo.category_variables + datainfo.boolean_variables + extended_vars
    # Directly compute the intersection of cat_vars and df.columns
    inc_vars = list(set(cat_vars) & set(df.columns))
    for c in inc_vars:
        df[c] = df[c].astype("category")
    return df


def load_kfold_indexes(
    df: pd.DataFrame, kfold_json_file: Path
) -> dict[str, dict[str, NDArray[np.int_]]]:
    """Load saved k-fold splits from JSON and convert IDs to dataframe indexes."""
    with open(kfold_json_file, "r") as f:
        saved_splits = json.load(f)

    index_splits: dict[str, dict[str, NDArray[np.int_]]] = {}
    for fold_name, split in saved_splits.items():
        train_idx = df.index[df["ID"].isin(split["train_ids"])].to_numpy()
        test_idx = df.index[df["ID"].isin(split["test_ids"])].to_numpy()
        index_splits[fold_name] = {"train_idx": train_idx, "test_idx": test_idx}
    return index_splits


def load_and_preprocess_data(
    train_file: Path,
    test_file: Path,
    kfold_json_file: Path,
    model_variant: str = "extended",
    extended_vars: Optional[list[str]] = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    dict[str, dict[str, NDArray[np.int_]]],
]:
    """
    Load and preprocess data for training/evaluation.

    Parameters
    ----------
    train_file : Path
        Path to training data CSV file
    test_file : Path
        Path to test data CSV file
    kfold_json_file : Path
        Path to k-fold splits JSON file
    model_variant : str
        Either 'original' (excludes extended variables) or 'extended' (includes all variables)
    extended_vars : list[str], optional
        List of additional variables to exclude in 'original' variant

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, index_splits
    """
    if model_variant not in ["original", "extended"]:
        raise ValueError(
            f"model_variant must be 'original' or 'extended', got '{model_variant}'"
        )

    if extended_vars is None:
        extended_vars = []

    # Load data
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    target = datainfo.target_lbr

    # Define variables to remove based on variant
    low_imp_vars = datainfo.LBR_low_imp
    if model_variant == "original":
        low_imp_vars = low_imp_vars + extended_vars
        print(
            f"[INFO] Loading 'original' data - excluding extended variables: {extended_vars}"
        )
    else:
        print(f"[INFO] Loading 'extended' data - including all variables")

    # Remove low importance variables
    df_train = remove_variables(df_train, low_imp_vars)
    df_test = remove_variables(df_test, low_imp_vars)

    # Set categorical variables
    df_train = set_cat_variables(df_train, extended_vars)
    df_test = set_cat_variables(df_test, extended_vars)

    # Prepare X and y
    X_train = df_train.drop(columns=[target, "ID"]).copy()
    y_train = df_train[target].copy()
    X_test = df_test.drop(columns=[target, "ID"]).copy()
    y_test = df_test[target].copy()

    # Load k-fold indexes
    index_splits = load_kfold_indexes(df_train, kfold_json_file)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(
        f"Train positive rate: {y_train.mean():.3f}, Test positive rate: {y_test.mean():.3f}\n"
    )

    return X_train, X_test, y_train, y_test, index_splits
