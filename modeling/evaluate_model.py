"""
Evaluate trained XGBoost models and generate visualizations.

This script loads a trained study, retrains the best model, and produces:
- ROC curves
- Feature importance rankings
- Performance metrics
"""

import argparse
from pathlib import Path
from typing import Any
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    root_str = str(project_root)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from modeling import data_loader, model_config as config, optuna_helper as oh


def evaluate_model(
    model_variant: str = "extended", save: bool = True, show_plots: bool = True
) -> dict[str, Any]:
    """
    Evaluate the best model from a study.

    Parameters
    ----------
    model_variant : str
        'original' or 'extended'
    save : bool
        Whether to save feature importance and plots
    show_plots : bool
        Whether to display plots
    """
    study_dir = config.STUDY_DIRS[model_variant]
    output_dir = config.MODEL_DIRS[model_variant]
    config.ensure_directories()

    # Load study
    study_file = oh.resolve_study_file("latest", study_dir)
    study = oh.load_study(study_file)

    print(f"========== EVALUATING BEST MODEL ({model_variant.upper()}) =============")
    print(f"Loaded from: {study_file.name}")
    print(f"Best value: {study.best_value:.5f}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Total trials: {len(study.trials)}\n")

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

    # Retrain with best parameters
    obj_kwargs = config.get_objective_kwargs()
    best_params = oh.update_best_model_params(study, obj_kwargs)
    kfold = StratifiedKFold(
        n_splits=config.N_SPLITS, random_state=config.RANDOM_STATE, shuffle=True
    )

    print("Retraining model with best parameters...")
    results = oh.fit_xgb_with_fixed_params(
        X_train,
        y_train,
        kfold,
        best_params,
        X_test=X_test,
        y_test=y_test,
        calc_shap_imp=True,
        return_shap_explainers=True,
        custom_split=index_splits,
    )

    # === Feature Importance ===
    feature_imp = results["importances"]
    df_feature_imp = pd.DataFrame(feature_imp, index=X_train.columns)
    df_feature_imp = df_feature_imp.sort_values(by="shap_values", ascending=False)

    print("\n=== TOP 10 FEATURES (SHAP) ===")
    print(df_feature_imp.head(10))

    if save:
        feature_imp_file = output_dir / "feature_importance.xlsx"
        df_feature_imp.to_excel(feature_imp_file)
        print(f"\nFeature importance saved to: {feature_imp_file}")

    # === Evaluation Metrics ===
    preds_train = results["eval"]["predictions"]["pred_proba_avg"]
    fpr_train, tpr_train, threshold_train = metrics.roc_curve(y_train, preds_train)
    auc_train = metrics.auc(fpr_train, tpr_train)

    preds_test = results["test"]["predictions"]["pred_proba_avg"]
    fpr_test, tpr_test, threshold_test = metrics.roc_curve(y_test, preds_test)
    auc_test = metrics.auc(fpr_test, tpr_test)

    # Optimal threshold (Youden's J statistic)
    opt_thres = threshold_train[np.argmax(tpr_train - fpr_train)]

    # Accuracies
    orig_train_acc = metrics.accuracy_score(y_train, preds_train > 0.5)
    opt_train_acc = metrics.accuracy_score(y_train, preds_train > opt_thres)
    orig_test_acc = metrics.accuracy_score(y_test, preds_test > 0.5)
    opt_test_acc = metrics.accuracy_score(y_test, preds_test > opt_thres)

    # Print results
    print(f"\n=== PERFORMANCE METRICS ({model_variant.upper()}) ===")
    print(f"Train AUC: {auc_train:.2f}")
    print(f"Test AUC:  {auc_test:.2f}")

    print(f"\n{'Threshold':<20} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 44)
    print(f"{'0.5 (default)':<20} {orig_train_acc:<12.4f} {orig_test_acc:<12.4f}")
    print(
        f"{f'{opt_thres:.4f} (optimal)':<20} {opt_train_acc:<12.4f} {opt_test_acc:<12.4f}"
    )

    print(f"\n--- Class Balance ---")
    print(f"Train: {y_train.mean():.3f} positive ({y_train.sum()}/{len(y_train)})")
    print(f"Test:  {y_test.mean():.3f} positive ({y_test.sum()}/{len(y_test)})")
    # === Visualizations ===
    if show_plots or save:
        # Combined ROC curve
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            fpr_train, tpr_train, "b", label=f"Train AUC = {auc_train:.2f}", linewidth=2
        )
        ax.plot(
            fpr_test, tpr_test, "g", label=f"Test AUC = {auc_test:.2f}", linewidth=2
        )
        ax.plot([0, 1], [0, 1], "r--", label="Random Classifier", linewidth=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve - {model_variant.capitalize()} Model", fontsize=14)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save:
            roc_file = output_dir / "roc_curve.png"
            plt.savefig(roc_file, dpi=300, bbox_inches="tight")
            print(f"ROC curve saved to: {roc_file}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate XGBoost model for LBR prediction"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="original",
        choices=["original", "extended"],
        help="Model variant to evaluate",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save feature importance and plots"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (useful for batch processing)",
    )

    args = parser.parse_args()

    evaluate_model(
        model_variant=args.variant, save=args.save, show_plots=not args.no_show
    )
