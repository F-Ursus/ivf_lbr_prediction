"""
XGBoost model training for live birth rate prediction.

Supports two model variants:
- 'original': Excludes extended variables (Cleavage_stage, Blastocyst_stage, FET)
- 'extended': Includes all variables
"""
import sys
import argparse
from pathlib import Path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    root_str = str(project_root)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from modeling import optuna_helper as oh, data_loader, model_config as config


def new_study(model_variant: str = "original", n_trials: int = 50) -> None:
    """
    Start a new hyperparameter optimization study.

    Parameters
    ----------
    model_variant : str
        'original' or 'extended'
    n_trials : int
        Number of trials to run
    """
    # Load data
    X_train, X_test, y_train, y_test, index_splits = (
        data_loader.load_and_preprocess_data(
            train_file=config.TRAIN_FILE,
            test_file=config.TEST_FILE,
            kfold_json_file=config.KFOLD_JSON_FILE,
            model_variant=model_variant,
            extended_vars=config.EXTENDED_VARS,
        )
    )

    # Run optimization
    obj_kwargs = config.get_objective_kwargs()
    study = oh.perform_study(X_train, y_train, obj_kwargs, n_trials=n_trials)

    print(f"\n=== STUDY COMPLETE ===")
    print(f"Best result: {study.best_value:.5f}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Total trials: {len(study.trials)}")

    # Save study
    config.ensure_directories()
    study_dir = config.STUDY_DIRS[model_variant]
    study_file = oh.gen_study_name("LBR", study, study_dir, study_name=model_variant)
    oh.save_study(study, study_file)
    print(f"Study saved to: {study_file}")


def continue_study(
    model_variant: str = "extended", n_trials: int = 100, n_iterations: int = 2
) -> None:
    """
    Continue an existing hyperparameter optimization study.

    Parameters
    ----------
    model_variant : str
        'original' or 'extended'
    n_trials : int
        Number of trials per iteration
    n_iterations : int
        Number of iterations to run
    """
    study_dir = config.STUDY_DIRS[model_variant]
    study_file = oh.resolve_study_file("latest", study_dir)
    study = oh.load_study(study_file)

    print(f"========== CONTINUING STUDY =============")
    print(f"Model variant: {model_variant}")
    print(f"Loaded from: {study_file.name}")

    if len(study.trials) > 0:
        print(f"Best value: {study.best_value:.5f}")
        print(f"Best trial: #{study.best_trial.number}")
        print(f"Total trials: {len(study.trials)}")
    print()

    # Load data
    X_train, X_test, y_train, y_test, index_splits = (
        data_loader.load_and_preprocess_data(
            train_file=config.TRAIN_FILE,
            test_file=config.TEST_FILE,
            kfold_json_file=config.KFOLD_JSON_FILE,
            model_variant=model_variant,
            extended_vars=config.EXTENDED_VARS,
        )
    )

    obj_kwargs = config.get_objective_kwargs()

    # Continue optimization
    for i in range(n_iterations):
        print(f"--- Iteration {i+1}/{n_iterations} ---")
        study = oh.continue_study(
            X_train,
            y_train,
            study,
            obj_kwargs,
            n_trials=n_trials,
            custom_splits=index_splits,
        )
        print(
            f"Best result: {study.best_value:.5f}, "
            f"Trial: {study.best_trial.number}, "
            f"Total: {len(study.trials)}"
        )

        study_file = oh.gen_study_name("LBR", study, study_dir)
        oh.save_study(study, study_file)
        print(f"Saved to: {study_file.name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost model for LBR prediction"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="original",
        choices=["original", "extended"],
        help="Model variant to train",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="new",
        choices=["new", "continue"],
        help="Start new study or continue existing",
    )
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of trials per iteration"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Number of iterations (for continue mode)",
    )

    args = parser.parse_args()

    if args.mode == "new":
        new_study(model_variant=args.variant, n_trials=args.trials)
    else:
        continue_study(
            model_variant=args.variant,
            n_trials=args.trials,
            n_iterations=args.iterations,
        )
