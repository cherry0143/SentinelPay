"""
Evaluation and Visualization Module.

Generates publication-quality charts and plots from computed metrics:
- Multi-model Confusion Matrices
- ROC Curves
- Precision-Recall Curves
- Performance Comparison Bar Charts
- Feature Importance Visualization
- Dataset Class Distribution
"""

import json
import logging
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configure aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def load_metrics(metrics_path: str = "reports/metrics.json") -> dict:
    """Load evaluation metrics JSON file."""
    with open(metrics_path, "r") as f:
        return json.load(f)


def plot_class_distribution(metrics: dict, output_dir: str = "reports"):
    """Plot dataset class distribution (legitimate vs suspicious proxy fraud)."""
    stats = metrics.get("label_stats", {})
    legit = stats.get("legitimate_count", 8681)
    fraud = stats.get("fraud_count", 269)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Legitimate (0)", "Suspicious / Fraud (1)"],
        [legit, fraud],
        color=["#2b5c8f", "#d9534f"],
        width=0.5,
    )
    ax.set_title("Credit Card Dataset Class Distribution (Severe Imbalance)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Accounts", fontsize=11)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h * 1.15,
            f"{int(h):,} ({h/(legit+fraud)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "class_distribution.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_confusion_matrices(metrics: dict, output_dir: str = "reports"):
    """Plot confusion matrices for the evaluated models side-by-side."""
    all_models = metrics.get("all_models", {})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    model_keys = ["Logistic Regression", "Random Forest", "HistGradientBoosting"]

    for idx, name in enumerate(model_keys):
        if name not in all_models:
            continue
        data = all_models[name]["tuned"]
        cm = np.array(data["confusion_matrix"])

        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if idx == 0 else "Greens" if idx == 1 else "Oranges",
            cbar=False,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(
            f"{name}\n(Recall: {data['recall']*100:.1f}%, F1: {data['f1_score']:.3f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xticklabels(["Normal (0)", "Fraud (1)"])
        ax.set_yticklabels(["Normal (0)", "Fraud (1)"])

    plt.suptitle("Held-Out Test Set Confusion Matrices (Threshold Tuned)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrices.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_roc_curves(metrics: dict, output_dir: str = "reports"):
    """Plot ROC curves for all models evaluated on test set."""
    y_test = np.array(metrics.get("test_ground_truth", []))
    probas = metrics.get("test_probabilities", {})

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, y_prob in probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = metrics["all_models"].get(name, {}).get("tuned", {}).get("roc_auc", 0.0)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (ROC-AUC = {auc_score:.4f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Classifier (0.50)")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "roc_curves.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_pr_curves(metrics: dict, output_dir: str = "reports"):
    """Plot Precision-Recall curves on held-out test data."""
    y_test = np.array(metrics.get("test_ground_truth", []))
    probas = metrics.get("test_probabilities", {})

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, y_prob in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = metrics["all_models"].get(name, {}).get("tuned", {}).get("pr_auc", 0.0)
        ax.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {pr_auc:.4f})")

    baseline = y_test.mean()
    ax.axhline(y=baseline, color="grey", linestyle="--", lw=1.5, label=f"No-Skill Baseline ({baseline*100:.1f}%)")
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Fraud Detected)", fontsize=11)
    ax.set_ylabel("Precision (True Fraud / Flagged)", fontsize=11)
    ax.set_title("Precision-Recall (PR) Curves for Fraud Detection", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "precision_recall_curves.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_model_comparison(metrics: dict, output_dir: str = "reports"):
    """Bar chart comparing all models across key metrics."""
    all_models = metrics.get("all_models", {})
    model_names = list(all_models.keys())

    f1_list = [all_models[m]["tuned"]["f1_score"] for m in model_names]
    rec_list = [all_models[m]["tuned"]["recall"] for m in model_names]
    prec_list = [all_models[m]["tuned"]["precision"] for m in model_names]
    pr_auc_list = [all_models[m]["tuned"]["pr_auc"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 1.5 * width, prec_list, width, label="Precision", color="#4285F4")
    ax.bar(x - 0.5 * width, rec_list, width, label="Recall", color="#EA4335")
    ax.bar(x + 0.5 * width, f1_list, width, label="F1-Score", color="#FBBC05")
    ax.bar(x + 1.5 * width, pr_auc_list, width, label="PR-AUC", color="#34A853")

    ax.set_title("Model Comparison on Held-Out Test Set", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, fontweight="bold")
    ax.set_ylim([0, 1.1])
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_feature_importance(metrics: dict, output_dir: str = "reports"):
    """Plot feature importance / coefficient magnitudes for best model."""
    fi = metrics.get("feature_importance", {})
    if not fi:
        logger.warning("No feature importance found in metrics")
        return

    sorted_fi = sorted(fi.items(), key=lambda item: item[1], reverse=True)
    features, weights = zip(*sorted_fi)

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, weights, color="#1a73e8", align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()  # Highest importance on top
    ax.set_xlabel("Relative Feature Weight / Importance Magnitude", fontsize=10)
    ax.set_title(f"Feature Importance ({metrics.get('best_model_name', 'Model')})", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def generate_all_plots(metrics_path: str = "reports/metrics.json", output_dir: str = "reports"):
    """Load metrics and generate all evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = load_metrics(metrics_path)
    logger.info("Generating evaluation visualizations...")
    plot_class_distribution(metrics, output_dir)
    plot_confusion_matrices(metrics, output_dir)
    plot_roc_curves(metrics, output_dir)
    plot_pr_curves(metrics, output_dir)
    plot_model_comparison(metrics, output_dir)
    plot_feature_importance(metrics, output_dir)
    logger.info("All plots generated successfully!")


if __name__ == "__main__":
    generate_all_plots()
