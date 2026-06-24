"""Evaluation: metrics, classification report, confusion matrix and plots.

Run:  python -m src.evaluate --config config.yaml \
            --model models/driver_distraction_model.keras

Produces, in outputs/:
  * metrics.json            (accuracy, macro precision/recall/F1)
  * classification_report.txt
  * confusion_matrix.png
  * per_class_f1.png
  * training_curves.png     (when called from train.py)
"""

import os
import json
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import tensorflow as tf

from src.utils import load_config, ensure_dir, get_class_labels
from src.data_loader import build_datasets


def plot_history(histories, outputs_dir):
    """Plot accuracy/loss curves across one or two training stages."""
    # Concatenate stage1 (+ stage2) so the curves read as one run.
    acc, val_acc, loss, val_loss = [], [], [], []
    for key in ("stage1", "stage2"):
        h = histories.get(key)
        if not h:
            continue
        acc += h.get("accuracy", [])
        val_acc += h.get("val_accuracy", [])
        loss += h.get("loss", [])
        val_loss += h.get("val_loss", [])

    epochs = range(1, len(acc) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, acc, "b-", label="train")
    axes[0].plot(epochs, val_acc, "r-", label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, loss, "b-", label="train")
    axes[1].plot(epochs, val_loss, "r-", label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(outputs_dir, "training_curves.png"), dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm, labels, outputs_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(outputs_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)


def plot_per_class_f1(per_class_f1, labels, outputs_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), per_class_f1, color="steelblue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1-score")
    fig.tight_layout()
    fig.savefig(os.path.join(outputs_dir, "per_class_f1.png"), dpi=120)
    plt.close(fig)


def evaluate(config_path="config.yaml", model_path=None):
    config = load_config(config_path)
    outputs_dir = ensure_dir(config["paths"]["outputs_dir"])
    labels = get_class_labels(config)

    if model_path is None:
        model_path = os.path.join(
            config["paths"]["models_dir"], "driver_distraction_model.keras"
        )
    model = tf.keras.models.load_model(model_path)

    # Use the validation split as the held-out evaluation set. Collect labels
    # and predictions in a single pass so they stay aligned even if the
    # dataset reshuffles between iterations.
    _, val_ds = build_datasets(config)
    y_true, y_pred = [], []
    for batch_x, batch_y in val_ds:
        batch_probs = model.predict(batch_x, verbose=0)
        y_true.append(batch_y.numpy())
        y_pred.append(np.argmax(batch_probs, axis=1))
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    # Pin every metric to the full label set so classes that are never
    # predicted still appear (otherwise sklearn silently drops them).
    label_ids = list(range(len(labels)))

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average="macro", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average=None, zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class_f1": {labels[i]: float(per_class_f1[i]) for i in range(len(labels))},
    }

    report = classification_report(
        y_true, y_pred, labels=label_ids, target_names=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    # ---- Persist everything ----------------------------------------------
    with open(os.path.join(outputs_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(outputs_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    plot_confusion_matrix(cm, labels, outputs_dir)
    plot_per_class_f1(per_class_f1, labels, outputs_dir)

    print("\n=== Evaluation summary ===")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro precision : {precision:.4f}")
    print(f"Macro recall    : {recall:.4f}")
    print(f"Macro F1-score  : {f1:.4f}")
    print("\n" + report)
    print(f"Saved metrics & plots to {outputs_dir}/")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate driver distraction CNN")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    evaluate(args.config, args.model)
