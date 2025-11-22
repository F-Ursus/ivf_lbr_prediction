"""
Create or verify k-fold cross-validation splits.

This script documents how kfold_ids.json was created to ensure reproducibility.
"""
import sys
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    root_str = str(project_root)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from data_process import datainfo

def create_kfold_splits(
    data_file: Path,
    output_file: Path,
    target_column: str = datainfo.target_lbr,
    id_column: str = "ID",
    n_splits: int = 5,
    random_state: int = 13,
    save: bool = True
) -> dict:
    """
    Create stratified k-fold splits and save patient IDs.
    
    Parameters
    ----------
    data_file : Path
        Path to training data CSV
    output_file : Path
        Path to save JSON with fold IDs
    target_column : str
        Name of target variable for stratification
    id_column : str
        Name of patient ID column
    n_splits : int
        Number of folds
    random_state : int
        Random seed for reproducibility
    save : bool
        Whether to save the splits to file
    
    Returns
    -------
    dict
        Dictionary with fold IDs
    """
    df = pd.read_csv(data_file)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_splits = {}
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df[target_column])):
        fold_name = f"fold_{fold_idx+1}"
        fold_splits[fold_name] = {
            "train_ids": df.iloc[train_idx][id_column].tolist(),
            "test_ids": df.iloc[test_idx][id_column].tolist()
        }
    
    if save:
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(fold_splits, f, indent=2)
        
        print(f"Created {n_splits} folds")
        print(f"Saved to: {output_file}")
        
        for fold_name, split in fold_splits.items():
            print(f"{fold_name}: {len(split['train_ids'])} train, {len(split['test_ids'])} test")
    
    return fold_splits


def verify_kfold_splits(splits_file: Path, data_file: Path, target_column: str):
    """Verify that existing splits are properly stratified."""
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    df = pd.read_csv(data_file)
    
    print("\n=== Verifying K-Fold Splits ===")
    for fold_name, split in splits.items():
        train_ids = split['train_ids']
        test_ids = split['test_ids']
        
        train_target = df[df['ID'].isin(train_ids)][target_column]
        test_target = df[df['ID'].isin(test_ids)][target_column]
        
        print(f"\n{fold_name}:")
        print(f"  Train: {len(train_ids)} samples, {train_target.mean():.3f} positive rate")
        print(f"  Test:  {len(test_ids)} samples, {test_target.mean():.3f} positive rate")


def compare_splits(existing_splits: dict, new_splits: dict) -> bool:
    """
    Compare two sets of k-fold splits to check if they're identical.
    
    Parameters
    ----------
    existing_splits : dict
        Existing splits loaded from file
    new_splits : dict
        Newly generated splits
    
    Returns
    -------
    bool
        True if splits are identical, False otherwise
    """
    print("\n=== Comparing Existing vs Newly Generated Splits ===")
    
    if set(existing_splits.keys()) != set(new_splits.keys()):
        print("ERROR: Different number of folds!")
        return False

    print(existing_splits.keys())
    print(new_splits.keys())
    
    all_match = True
    for fold_name in existing_splits.keys():
        existing_train = set(existing_splits[fold_name]['train_ids'])
        existing_test = set(existing_splits[fold_name]['test_ids'])
        
        new_train = set(new_splits[fold_name]['train_ids'])
        new_test = set(new_splits[fold_name]['test_ids'])
        
        train_match = existing_train == new_train
        test_match = existing_test == new_test
        
        if train_match and test_match:
            print(f"MATCH: {fold_name} - IDENTICAL")
        else:
            print(f"DIFFER: {fold_name} - DIFFERENT")
            if not train_match:
                print(f"   Train IDs differ by {len(existing_train.symmetric_difference(new_train))} samples")
            if not test_match:
                print(f"   Test IDs differ by {len(existing_test.symmetric_difference(new_test))} samples")
            all_match = False
    
    if all_match:
        print("\nSUCCESS: Existing splits can be reproduced exactly!")
        print("The kfold_ids.json file was created with:")
        print("  - StratifiedKFold(n_splits=5, shuffle=True, random_state=13)")
    else:
        print("\nWARNING: Splits differ! Either:")
        print("  - Different random_state was used")
        print("  - Different data order in the CSV")
        print("  - Different sklearn version")
    
    return all_match


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data"
    
    train_file = data_path / "LBR_train.csv"
    splits_file = data_path / "kfold_ids.json"
    
    # Check if train file exists
    if not train_file.exists():
        print(f"ERROR: Training data not found at {train_file}")
        print("Please run preprocess_data.py first to create LBR_train.csv")
        sys.exit(1)
    
    # Check if splits already exist
    if not splits_file.exists():
        print("No existing kfold_ids.json found.")
        print("Creating new k-fold splits...")
        create_kfold_splits(
            data_file=train_file,
            output_file=splits_file,
            target_column=datainfo.target_lbr,
            n_splits=5,
            random_state=13,
            save=True
        )
        verify_kfold_splits(splits_file, train_file, target_column=datainfo.target_lbr)
        print("\nK-fold splits created successfully.")
        
    else:
        print("Existing kfold_ids.json found.")
        
        # Load existing splits
        with open(splits_file, 'r') as f:
            existing_splits = json.load(f)
        
        # Verify existing splits
        verify_kfold_splits(splits_file, train_file, target_column=datainfo.target_lbr)
        
        # Ask user what to do
        print("\nOptions:")
        print("  1. Keep existing splits (recommended)")
        print("  2. Verify reproducibility (generate new splits and compare)")
        print("  3. Recreate splits (WARNING: this might change all results!)")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            print("Keeping existing splits.")
            
        elif choice == "2":
            # Generate new splits (without saving)
            print("\nGenerating new splits with same parameters...")
            new_splits = create_kfold_splits(
                data_file=train_file,
                output_file=splits_file,
                target_column=datainfo.target_lbr,
                n_splits=5,
                random_state=13,
                save=False
            )
            
            # Compare
            splits_match = compare_splits(existing_splits, new_splits)
            
            if splits_match:
                print("\nExisting splits are reproducible. No action needed.")
            else:
                print("\nExisting splits cannot be reproduced with current settings.")
                recreate = input("Recreate splits? (yes/no): ").strip().lower()
                if recreate == 'yes':
                    create_kfold_splits(
                        data_file=train_file,
                        output_file=splits_file,
                        target_column=datainfo.target_lbr,
                        n_splits=5,
                        random_state=13,
                        save=True
                    )
                    print("Splits file recreated.")
                    
        elif choice == "3":
            confirm = input("Are you sure? This will invalidate all previous results! (yes/no): ").strip().lower()
            if confirm == 'yes':
                create_kfold_splits(
                    data_file=train_file,
                    output_file=splits_file,
                    target_column=datainfo.target_lbr,
                    n_splits=5,
                    random_state=13,
                    save=True
                )
                print("Splits file recreated.")
            else:
                print("Cancelled. Keeping existing splits.")
        else:
            print("Invalid choice. Keeping existing splits.")