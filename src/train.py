"""
Model Training, Cross-Validation, and Comparison Pipeline.

Trains and evaluates:
1. Logistic Regression (Interpretable baseline with class weighting)
2. Random Forest (Nonlinear bagging ensemble)
3. HistGradientBoosting (Fast gradient boosting classifier)
4. Isolation Forest (Unsupervised anomaly detection baseline)

Selects the best model based on fraud-specific metrics (PR-AUC, F1, Recall),
optimizes classification thresholds, and persists model artifacts.
"""

import json
import logging
import os
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data_processing import (
    apply_resampling,
    create_preprocessor,
    generate_proxy_fraud_labels,
    load_raw_data,
    prepare_features_and_target,
    profile_dataset,
    split_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find threshold that optimizes F1 score on validation/training probabilities.
    Prevents test set leakage by selecting threshold strictly prior to test evaluation.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Avoid division by zero
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * (precisions * recalls) / (precisions + recalls + 1e-10),
        0.0,
    )
    # thresholds array has length len(precisions) - 1
    best_idx = np.argmax(f1_scores[:-1])
    best_thresh = float(thresholds[best_idx])
    best_score = float(f1_scores[best_idx])
    return best_thresh, best_score


def evaluate_model(
    name: str,
    y_test: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate comprehensive fraud detection metrics at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_test, y_proba))
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_test, y_pred).tolist()
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    metrics = {
        "model_name": name,
        "threshold": round(threshold, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }
    return metrics


def train_and_evaluate_all(
    dataset_path: str = "CC GENERAL.csv",
    output_dir: str = "models",
    reports_dir: str = "reports",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Full pipeline to train, evaluate, compare, and persist the best model.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load and Profile Raw Data
    df = load_raw_data(dataset_path)
    profile = profile_dataset(df)

    # 2. Unsupervised Proxy Fraud Labeling
    df_labeled, iso_model, label_stats = generate_proxy_fraud_labels(
        df, contamination=0.03, random_state=random_state
    )

    # 3. Prepare Features & Target
    X, y, feature_names = prepare_features_and_target(df_labeled, target_col="is_fraud")

    # 4. Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=random_state
    )

    # 5. Fit Preprocessor STRICTLY on X_train
    preprocessor = create_preprocessor(feature_names)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # 6. Apply SMOTE strictly on Training Set
    X_train_resampled, y_train_resampled = apply_resampling(
        X_train_trans, y_train, method="smote", random_state=random_state
    )

    # 7. Model Candidates Setup
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=random_state,
        ),
    }

    results = {}
    fitted_models = {}
    optimal_thresholds = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    logger.info("=== Starting Cross-Validation and Training for Supervised Models ===")
    for name, model in models.items():
        logger.info("Training %s...", name)

        # Tune threshold via Out-Of-Fold CV probabilities on training set
        cv_probas = cross_val_predict(
            model,
            X_train_resampled,
            y_train_resampled,
            cv=cv,
            method="predict_proba",
        )[:, 1]

        opt_thresh, opt_f1 = find_optimal_threshold(y_train_resampled, cv_probas)
        # Cap threshold within practical range
        opt_thresh = max(0.20, min(opt_thresh, 0.80))
        optimal_thresholds[name] = opt_thresh

        # Fit on full resampled training set
        model.fit(X_train_resampled, y_train_resampled)
        fitted_models[name] = model

        # Evaluate on untouched test set
        test_probas = model.predict_proba(X_test_trans)[:, 1]

        # Standard 0.5 threshold evaluation
        res_default = evaluate_model(name, y_test.to_numpy(), test_probas, threshold=0.5)
        # Tuned threshold evaluation
        res_tuned = evaluate_model(
            f"{name} (Tuned Thresh)",
            y_test.to_numpy(),
            test_probas,
            threshold=opt_thresh,
        )

        results[name] = {
            "default_metrics": res_default,
            "tuned_metrics": res_tuned,
            "test_probas": test_probas.tolist(),
        }

        logger.info(
            "%s -> Default F1: %.4f, Recall: %.4f, PR-AUC: %.4f | Tuned (th=%.2f) F1: %.4f, Recall: %.4f",
            name,
            res_default["f1_score"],
            res_default["recall"],
            res_default["pr_auc"],
            opt_thresh,
            res_tuned["f1_score"],
            res_tuned["recall"],
        )

    # 8. Evaluate Isolation Forest as an Unsupervised Anomaly Detection Baseline
    logger.info("Evaluating Isolation Forest baseline on test set...")
    iso_test_scores = -iso_model.decision_function(
        create_preprocessor(feature_names).fit_transform(X_test)
    )
    # Normalize anomaly scores to [0, 1] range for metric consistency
    iso_norm = (iso_test_scores - iso_test_scores.min()) / (
        iso_test_scores.max() - iso_test_scores.min() + 1e-10
    )
    iso_metrics = evaluate_model(
        "Isolation Forest (Unsupervised)",
        y_test.to_numpy(),
        iso_norm,
        threshold=np.percentile(iso_norm, 97),
    )
    results["Isolation Forest"] = {
        "default_metrics": iso_metrics,
        "tuned_metrics": iso_metrics,
        "test_probas": iso_norm.tolist(),
    }

    # 9. Model Selection Strategy:
    # In fraud detection, we prioritize PR-AUC and F1-score with high Recall.
    # We compare the tuned metrics of candidate supervised models.
    best_name = None
    best_score = -1.0
    for name in models.keys():
        f1 = results[name]["tuned_metrics"]["f1_score"]
        pr_auc = results[name]["tuned_metrics"]["pr_auc"]
        composite = 0.5 * f1 + 0.5 * pr_auc
        if composite > best_score:
            best_score = composite
            best_name = name

    best_model = fitted_models[best_name]
    best_thresh = optimal_thresholds[best_name]
    best_metrics = results[best_name]["tuned_metrics"]

    logger.info(
        "*** SELECTED BEST MODEL: %s (Composite Score: %.4f, F1: %.4f, Recall: %.4f, PR-AUC: %.4f) ***",
        best_name,
        best_score,
        best_metrics["f1_score"],
        best_metrics["recall"],
        best_metrics["pr_auc"],
    )

    # 10. Compute Feature Importances for the Best Model
    feature_importance_dict = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importance_dict = {
            feat: float(imp) for feat, imp in zip(feature_names, importances)
        }
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
        feature_importance_dict = {
            feat: float(imp) for feat, imp in zip(feature_names, importances)
        }

    # 11. Persist Model Bundle
    artifact = {
        "model_name": best_name,
        "model": best_model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "classification_threshold": best_thresh,
        "test_metrics": best_metrics,
        "all_models_summary": {
            k: {
                "default": v["default_metrics"],
                "tuned": v["tuned_metrics"],
            }
            for k, v in results.items()
        },
        "feature_importance": feature_importance_dict,
        "dataset_profile": profile,
        "label_stats": label_stats,
        "training_metadata": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_fraud_count": int(y_train.sum()),
            "test_fraud_count": int(y_test.sum()),
            "smote_train_samples": len(X_train_resampled),
            "resampling_strategy": "SMOTE (train split only)",
            "scaling_strategy": "RobustScaler with Median SimpleImputer",
            "random_state": random_state,
        },
    }

    model_save_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump(artifact, model_save_path)
    logger.info("Saved best model artifact bundle to %s", model_save_path)

    # 12. Persist Metrics Report to JSON (excluding non-serializable objects)
    metrics_report = {
        "best_model_name": best_name,
        "classification_threshold": best_thresh,
        "best_metrics": best_metrics,
        "all_models": {
            k: {
                "default": v["default_metrics"],
                "tuned": v["tuned_metrics"],
            }
            for k, v in results.items()
        },
        "feature_importance": feature_importance_dict,
        "dataset_profile": profile,
        "label_stats": label_stats,
        "test_ground_truth": y_test.tolist(),
        "test_probabilities": {
            k: v["test_probas"] for k, v in results.items()
        },
    }

    metrics_save_path = os.path.join(reports_dir, "metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    logger.info("Saved evaluation report to %s", metrics_save_path)

    return artifact


if __name__ == "__main__":
    train_and_evaluate_all()
