# compare_excels.py
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    root_str = str(project_root)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

import data_process.datainfo as datainfo


def load_excel(path: Path, sheet: str | int | None) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    except ImportError as e:
        raise SystemExit(
            "Missing dependency to read .xlsx. Install with:\n  pip install pandas openpyxl"
        ) from e


def initial_cleaning(data_path, file_name, save=False):
    """
    Performs initial cleaning of the dataset:
    - Loads Excel file
    - Fixes known wrong values
    - Removes unnecessary columns
    - Drops binary variables with low counts
    - Fills missing values for certain features
    - Optionally saves cleaned file

    Parameters
    ----------
    data_path : Path | str
        Path to folder containing the file.
    file_name : str
        Excel file name to load.
    save : bool, optional
        Whether to save the cleaned dataset as a new Excel file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    print("=== Running initial_cleaning() ===")
    # === Load dataset ===
    df = pd.read_excel(Path(data_path).joinpath(file_name), sheet_name=0)
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # === Fix known missing / incorrect outcome ===
    lbr_target = datainfo.target_lbr
    df.loc[df["ID"] == 1129, lbr_target] = 0

    # === Fix known incorrect values ===
    df.loc[df["Previous_child"] == 2, "Previous_child"] = 1
    df.loc[
        df["Previous_spontaneous_abortion"] == 2, "Previous_spontaneous_abortion"
    ] = 1

    # === Remove specific unnecessary columns ===
    cols_to_remove = [
        "Largest_diameter_of_lesion_1",  # too many missing values
        "A_Location_2D",
        "Uterine_location_3D",  # bundled into Location_tot
        "A_Extent_2D",
        "JZ_Extent_irregularity_3D",  # bundled into Extent
    ]
    print(f"Shape before removing fixed columns: {df.shape}")
    df_removed = df.drop(columns=cols_to_remove)
    print(f"Shape after removing fixed columns: {df_removed.shape}")

    # === Remove binary variables with too few positive cases ===
    df_removed = df_removed.drop(columns=datainfo.actual_removals)
    print(f"Shape after removing low-count binary variables: {df_removed.shape}")

    # === Fill missing values ===
    df_removed["Extent"] = df_removed["Extent"].fillna(0)
    df_removed["Location_tot"] = df_removed["Location_tot"].fillna(0)
    print("Filled missing values in 'Extent' and 'Location_tot' with 0.")

    # === Sanity checks ===
    total_missing = df_removed.isna().sum().sum()
    if total_missing > 0:
        print(f"[Warning]: {total_missing} missing values remain after cleaning.")
    else:
        print("[OK] No remaining missing values after cleaning.")

    # Check for duplicate IDs
    duplicate_ids = df_removed["ID"].duplicated().sum()
    if duplicate_ids > 0:
        print(f"[Warning]: {duplicate_ids} duplicate IDs found in dataset.")
    else:
        print("[OK] No duplicate IDs found.")

    # === Save cleaned dataset if requested ===
    if save:
        xlsx_out_fixed = Path(data_path).joinpath(
            file_name.split(".")[0] + " - fixed.xlsx"
        )
        df_removed.to_excel(xlsx_out_fixed, index=False)
        print(f"Cleaned dataset saved to: {xlsx_out_fixed}")

    print(
        f"Final cleaned dataset: {df_removed.shape[0]} rows × {df_removed.shape[1]} columns\n"
    )

    return df_removed


def train_test_split_wrapper(new_df, data_path, save=False):
    print("=== Running train_test_split ===")
    RANDOM_STATE = 13
    df_train, df_test = train_test_split(
        new_df,
        test_size=0.20,
        stratify=new_df[datainfo.target_lbr],
        random_state=RANDOM_STATE,
    )
    df_train.sort_values(by="ID", inplace=True)
    df_test.sort_values(by="ID", inplace=True)

    # Sanity checks
    print("=== Shape ===")
    print(f"df train shape: {df_train.shape}")
    print(f"df test  shape: {df_test.shape}")

    if save:
        df_train.to_excel(data_path.joinpath("LBR_train.xlsx"), index=False)
        df_test.to_excel(data_path.joinpath("LBR_test.xlsx"), index=False)
        df_train.to_csv(data_path.joinpath("LBR_train.csv"), index=False)
        df_test.to_csv(data_path.joinpath("LBR_test.csv"), index=False)
        print(f"Saved train/test splits in both .xlsx and .csv formats to: {data_path}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data"
    file_name = "XGBoost_Adeno_ART_LBR_250823.xlsx"

    # Check if input file exists
    input_file = data_path / file_name
    if not input_file.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {input_file}\n"
            f"Please ensure '{file_name}' is in the data/ folder."
        )

    print(f"Processing data from: {input_file}")
    df = initial_cleaning(data_path, file_name, save=True)
    train_test_split_wrapper(df, data_path, save=True)


if __name__ == "__main__":
    main()
