from datetime import datetime

import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from optuna import create_study
from optuna.samplers import TPESampler
import shap
from sklearn.impute import SimpleImputer
import pandas as pd
import pickle
import re
from pathlib import Path
from typing import Any


def objective(
    trial,
    X,
    y,
    random_state=13,
    n_splits=5,
    n_repeats=5,
    n_jobs=1,
    early_stopping_rounds=10,
    model_eval_metric="avg_acc",
    custom_split=None,
) -> float:
    """Objective function for Optuna hyperparameter optimization."""

    params = {
        "verbosity": 0,  # 0 (silent) - 3 (debug)
        "objective": "binary:logistic",
        "tree_method": "hist",
        "n_estimators": trial.suggest_int("n_estimators", 499, 500),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 2, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1, log=True),
        "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 4, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 4, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.2, 10, log=True),
        "seed": random_state,
        "n_jobs": n_jobs,
        "use_label_encoder": False,
        "eval_metric": "auc",
        "enable_categorical": True,
        "early_stopping_rounds": early_stopping_rounds,
    }

    rkf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    results = fit_xgb_with_fixed_params(X, y, rkf, params, custom_split=custom_split)
    model_perf = results["eval"]["metrics"][model_eval_metric]
    return model_perf


def fit_xgb_with_fixed_params(
    X,
    y,
    rkf,
    hp,
    X_test=None,
    y_test=None,
    calc_shap_imp=False,
    return_shap_explainers=False,
    custom_split=None,
) -> dict[str, Any]:
    model = xgb.XGBClassifier(**hp)
    X_values = X.copy()
    y_values = y.copy()
    y_pred = np.zeros_like(y_values)
    y_pred_proba = np.zeros_like(y_values * 1.0)
    accs = []
    aucs = []
    test_set = False
    importances = np.zeros((X.shape[1],))
    shap_importances = np.zeros((X.shape[1],))
    shap_expaliners = []
    foo = 0
    if X_test is not None and y_test is not None:
        X_test_values = X_test.copy()
        test_accs = []
        test_aucs = []
        test_pred = np.zeros_like(y_test)
        test_pred_proba = np.zeros_like(y_test * 1.0)
        test_set = True

    # Maybe add a counter and if index_split is not none override train_index and test_index isnide this forloop?
    k_counter = 1
    for train_index, test_index in rkf.split(X_values, y_values):
        if custom_split:
            train_index = custom_split[f"fold_{k_counter}"]["train_idx"]
            test_index = custom_split[f"fold_{k_counter}"]["test_idx"]
            k_counter += 1

        X_A, X_B = X_values.iloc[train_index], X_values.iloc[test_index]
        y_A, y_B = y_values.iloc[train_index], y_values.iloc[test_index]

        categorical_imputer, numerical_imputer = fit_imputer(
            X_A, categorical_strategy="most_frequent", numerical_strategy="mean"
        )

        X_A_imputed = apply_imputer(X_A, categorical_imputer, numerical_imputer)
        X_B_imputed = apply_imputer(X_B, categorical_imputer, numerical_imputer)
        # Ensure keep original order
        X_A_imputed = X_A_imputed[X_A.columns]
        X_B_imputed = X_B_imputed[X_B.columns]

        model.fit(
            X_A_imputed,
            y_A,
            eval_set=[(X_B_imputed, y_B)],
            verbose=0
        )

        # Get prediction and metrics
        curr_preds = model.predict(
            X_B_imputed, iteration_range=(0, model.best_iteration + 1)
        )
        curr_preds_proba = model.predict_proba(
            X_B_imputed, iteration_range=(0, model.best_iteration + 1)
        )[:, 1]
        y_pred[test_index] += curr_preds
        y_pred_proba[test_index] += curr_preds_proba

        accs.append(metrics.accuracy_score(y_true=y_B, y_pred=curr_preds))
        aucs.append(metrics.roc_auc_score(y_true=y_B, y_score=curr_preds_proba))
        # Feature importances
        importances += model.feature_importances_

        if calc_shap_imp:
            # Shap values
            foo += 1
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_B_imputed)

            shap_importances += shap_values.abs.mean(axis=0).values

            if return_shap_explainers:
                shap_expaliners.append(shap_values)

        if test_set:
            # If test set is given, calculate the predictions as well.
            X_test_values_imp = apply_imputer(
                X_test_values, categorical_imputer, numerical_imputer
            )
            # Ensure correct order of variables
            X_test_values_imp = X_test_values_imp[X_test_values.columns]
            curr_test_preds = model.predict(
                X_test_values_imp, iteration_range=(0, model.best_iteration + 1)
            )
            curr_test_preds_proba = model.predict_proba(
                X_test_values_imp, iteration_range=(0, model.best_iteration + 1)
            )[:, 1]
            test_pred += curr_test_preds
            test_pred_proba += curr_test_preds_proba
            test_accs.append(
                metrics.accuracy_score(y_true=y_test, y_pred=curr_test_preds)
            )
            test_aucs.append(
                metrics.roc_auc_score(y_true=y_test, y_score=curr_test_preds_proba)
            )

    # Get final results
    y_pred_avg = y_pred
    y_pred_prob_avg = y_pred_proba
    # Define result dict
    result_dict: dict[str, Any] = {
        "eval": create_result_dict(y, y_pred_avg, y_pred_prob_avg, accs, aucs)
    }
    if test_set:
        # If test set is given, calculate needed
        test_pred_avg = test_pred / rkf.get_n_splits()
        test_pred_proba_avg = test_pred_proba / rkf.get_n_splits()
        result_dict.update(
            {
                "test": create_result_dict(
                    y_test, test_pred_avg, test_pred_proba_avg, test_accs, test_aucs
                )
            }
        )
    try:
        eps = 0.0001
        result_dict.update(
            {
                "importances": {
                    "value": importances,
                    "value_norm": importances / (np.sum(importances) + eps),
                    "shap_values": shap_importances,
                    "shap_values_norm": shap_importances
                    / (np.sum(shap_importances) + eps),
                },
                "explainers": shap_expaliners,
            }
        )

    except Exception as err:
        print(f"Importances: {importances}")
        print(f"Sum importances: {np.sum(importances)}")
        print(f"Best n_tree_limit: {model.best_iteration + 1}")
        print(result_dict)
        print(err)
    return result_dict


def create_result_dict(y, y_pred_avg, y_pred_prob_avg, accs, aucs) -> dict[str, Any]:
    """Create the result dictionary from the different prediction inputs"""
    result_dict = {
        "metrics": {
            "ensemble_avg_acc": metrics.accuracy_score(
                y_true=y, y_pred=y_pred_avg > 0.5
            ),
            "ensemble_avg_prob_acc": metrics.accuracy_score(
                y_true=y, y_pred=y_pred_prob_avg > 0.5
            ),
            "ensemble_avg_auc": metrics.roc_auc_score(y_true=y, y_score=y_pred_avg),
            "ensemble_avg_prob_auc": metrics.roc_auc_score(
                y_true=y, y_score=y_pred_prob_avg
            ),
            "avg_acc": np.mean(accs),
            "avg_auc": np.mean(aucs),
            "accs": accs,
            "acc_std": np.std(accs),
            "aucs": aucs,
            "auc_std": np.std(aucs),
        },
        "predictions": {"pred_avg": y_pred_avg, "pred_proba_avg": y_pred_prob_avg},
    }
    return result_dict


def get_nth_best_params(study: optuna.Study, n: int = 2) -> dict[str, Any]:
    """
    Returns the n-th best hyperparameters from an Optuna study.

    Parameters
    ----------
    study : optuna.Study
        The Optuna study object.
    n : int, optional (default=2)
        The rank of the trial to fetch (1 = best, 2 = second-best, etc.).

    Returns
    -------
    Dict
        A dictionary containing the hyperparameters of the n-th best trial.
    """

    # Get only completed trials
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed_trials) < n:
        raise ValueError(
            f"Not enough completed trials to fetch the {n}-th best parameters."
        )

    # Sort trials by their objective values
    # (reverse=True for maximization, False for minimization)
    is_maximizing = study.direction == optuna.study.StudyDirection.MAXIMIZE
    sorted_trials = sorted(
        completed_trials, key=lambda t: t.value, reverse=is_maximizing
    )

    # Select the n-th best trial
    nth_best_trial = sorted_trials[n - 1]
    params = nth_best_trial.params

    # Print results
    print("=" * 60)
    print(f"{n}-th Best Hyperparameters")
    print("=" * 60)
    for k, v in params.items():
        print(f"{k}: {v}")
    print(f"\nObjective value: {nth_best_trial.value:.4f}")
    print("=" * 60)

    return params


def update_best_model_params(study, obj_kwargs) -> dict[str, Any]:
    # hp = study.best_params

    hp = get_nth_best_params(study, 1)
    hp["n_estimators"] = 500
    hp["verbosity"] = 0
    hp["objective"] = "binary:logistic"
    hp["seed"] = obj_kwargs["random_state"]
    hp["n_jobs"] = obj_kwargs["n_jobs"]
    hp["tree_method"] = "approx"
    hp["eval_metric"] = "auc"
    hp["early_stopping_rounds"] = obj_kwargs["early_stopping_rounds"]
    hp["enable_categorical"] = True

    return hp


def perform_study(X, y, obj_kwargs, n_trials=50, multivariate=False) -> optuna.Study:
    sampler = TPESampler(seed=obj_kwargs["random_state"], multivariate=multivariate)
    study = create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, X, y, **obj_kwargs),
        n_trials=n_trials,
        n_jobs=1,
    )

    return study


def continue_study(
    X, y, study, obj_kwargs, n_trials=5, custom_splits=None
) -> optuna.Study:
    study.optimize(
        lambda trial: objective(trial, X, y, custom_split=custom_splits, **obj_kwargs),
        n_trials=n_trials,
        n_jobs=1,
    )
    return study


def fit_imputer(
    dataframe, categorical_strategy="most_frequent", numerical_strategy="mean"
) -> tuple[SimpleImputer, SimpleImputer]:
    """
    Fit SimpleImputer on the training set for imputing missing values.

    Parameters:
    - dataframe: pandas DataFrame (training set).
    - categorical_strategy: Strategy to impute missing values for categorical variables.
                             Options: 'most_frequent', 'mode'.
    - numerical_strategy: Strategy to impute missing values for numerical variables.
                           Options: 'mean', 'median'.

    Returns:
    - categorical_imputer: Fitted SimpleImputer for categorical variables.
    - numerical_imputer: Fitted SimpleImputer for numerical variables.
    """

    # Separate columns by data type
    categorical_columns = dataframe.select_dtypes(include="category").columns
    numerical_columns = dataframe.select_dtypes(include="number").columns

    # Initialize SimpleImputer objects for categorical and numerical variables
    categorical_imputer = SimpleImputer(strategy=categorical_strategy)
    numerical_imputer = SimpleImputer(strategy=numerical_strategy)

    # Fit the imputers on the respective columns
    categorical_imputer.fit(dataframe[categorical_columns])
    numerical_imputer.fit(dataframe[numerical_columns])
    # print(categorical_columns)
    # print(numerical_columns)
    return categorical_imputer, numerical_imputer


def apply_imputer(dataframe, categorical_imputer, numerical_imputer) -> pd.DataFrame:
    """
    Apply fitted SimpleImputer on a DataFrame to impute missing values.

    Parameters:
    - dataframe: pandas DataFrame (training or validation set).
    - categorical_imputer: Fitted SimpleImputer for categorical variables.
    - numerical_imputer: Fitted SimpleImputer for numerical variables.

    Returns:
    - imputed_dataframe: DataFrame with missing values imputed.
    """

    # Separate columns by data type
    categorical_columns = dataframe.select_dtypes(include="category").columns
    numerical_columns = dataframe.select_dtypes(include="number").columns

    # Transform the imputers on the respective columns
    imputed_categorical = pd.DataFrame(
        categorical_imputer.transform(dataframe[categorical_columns]),
        columns=categorical_columns,
    )
    imputed_numerical = pd.DataFrame(
        numerical_imputer.transform(dataframe[numerical_columns]),
        columns=numerical_columns,
    )

    # Concatenate the imputed categorical and numerical columns
    imputed_dataframe = pd.concat([imputed_categorical, imputed_numerical], axis=1)

    return imputed_dataframe


def gen_datetime_string() -> str:
    now = datetime.now()
    return now.strftime("%m_%d_%Y")


def save_study(study, filepath) -> None:
    ### saves a study to be loaded later
    with open(filepath, "wb") as f:
        pickle.dump(study, f)


def gen_study_name(data_name, study, study_dir, study_name=None) -> Path:
    if study_name is None:
        study_file = study_dir.joinpath(
            f"{data_name}_STUDY_{gen_datetime_string()}_{len(study.trials)}_trials.pickle"
        )
    else:
        study_file = study_dir.joinpath(
            f"{data_name}_{study_name}_{gen_datetime_string()}_{len(study.trials)}_trials.pickle"
        )
    return study_file


def load_study(filepath) -> optuna.Study:
    with open(filepath, "rb") as f:
        study = pickle.load(f)

    return study


def latest_trials_file(study_dir: Path, glob: str = "*_trials.*") -> Path:
    """
    Find the study file with the highest trial number in its filename.

    Searches for files matching the pattern `*_<number>_trials.*` and returns
    the one with the largest number.

    Parameters
    ----------
    study_dir : Path
        Directory to search for trial files
    glob : str, default="*_trials.*"
        Glob pattern to match files

    Returns
    -------
    Path
        Path to the file with the highest trial number
    """
    # Regex to extract trial number from filename pattern: *_<number>_trials.*
    num_pattern = re.compile(r"_(\d+)_trials(?:\.\w+)?$")

    matching_files = []

    for filepath in study_dir.glob(glob):
        match = num_pattern.search(filepath.name)
        if match:
            trial_num = int(match.group(1))
            matching_files.append((trial_num, filepath))

    if not matching_files:
        raise FileNotFoundError(
            f"No files matching pattern '{glob}' found in {study_dir}"
        )

    # Sort by trial number (descending) and return the file with highest number
    matching_files.sort(reverse=True, key=lambda x: x[0])
    return matching_files[0][1]


def resolve_study_file(name_or_latest: str, study_dir: Path) -> Path:
    """
    Resolve study file path - either 'latest' or explicit path.

    Parameters
    ----------
    name_or_latest : str
        Either 'latest' to get the most recent trial file, or a file path
    study_dir : Path
        Directory containing study files

    Returns
    -------
    Path
        Resolved file path
    """
    if name_or_latest == "latest":
        return latest_trials_file(study_dir)
    p = Path(name_or_latest)
    return p if p.is_absolute() else study_dir / p
