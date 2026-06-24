"""Training pipeline.

Run:  python -m src.train --config config.yaml

Pipeline stages:
  1. Build augmented train/val generators.
  2. Build the model selected in config (custom CNN or transfer model).
  3. Stage 1 — train with a frozen backbone.
  4. Stage 2 — optionally unfreeze top layers and fine-tune at a low LR.
  5. Save the best model + the training-history plots.

Callbacks give us checkpointing (best val accuracy), early stopping
(stop when validation stops improving — an overfitting guard) and a
learning-rate scheduler (drop the LR on plateaus for finer convergence).
"""

import os
import json
import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

from src.utils import load_config, ensure_dir
from src.data_loader import build_datasets, compute_class_weights
from src.model import build_model, unfreeze_for_fine_tuning
from src.evaluate import plot_history


def _callbacks(config, stage_tag):
    models_dir = ensure_dir(config["paths"]["models_dir"])
    t = config["training"]
    ckpt_path = os.path.join(models_dir, f"best_model_{stage_tag}.keras")
    return [
        ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=t["early_stopping_patience"],
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=t["reduce_lr_factor"],
            patience=t["reduce_lr_patience"], min_lr=t["min_lr"], verbose=1,
        ),
    ], ckpt_path


def train(config_path="config.yaml"):
    config = load_config(config_path)
    tf.random.set_seed(config["data"]["seed"])

    outputs_dir = ensure_dir(config["paths"]["outputs_dir"])
    models_dir = ensure_dir(config["paths"]["models_dir"])

    # ---- Data -------------------------------------------------------------
    train_ds, val_ds = build_datasets(config)
    class_weights = compute_class_weights(config)
    print(f"Class weights: {class_weights}")

    # ---- Model ------------------------------------------------------------
    model = build_model(config)
    t = config["training"]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=t["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ---- Stage 1: frozen backbone ----------------------------------------
    cbs, _ = _callbacks(config, "stage1")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=t["epochs"],
        callbacks=cbs,
        class_weight=class_weights,
    )
    histories = {"stage1": history.history}

    # ---- Stage 2: fine-tuning (transfer models only) ---------------------
    if config["model"].get("fine_tune") and config["model"]["architecture"] != "custom_cnn":
        print("\n=== Fine-tuning: unfreezing top backbone layers ===")
        model = unfreeze_for_fine_tuning(model, config["model"]["fine_tune_at"])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=t["fine_tune_learning_rate"]
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        cbs2, _ = _callbacks(config, "stage2")
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=t["fine_tune_epochs"],
            callbacks=cbs2,
            class_weight=class_weights,
        )
        histories["stage2"] = history_ft.history

    # ---- Save final model + history --------------------------------------
    final_path = os.path.join(models_dir, "driver_distraction_model.keras")
    model.save(final_path)
    print(f"\nSaved final model -> {final_path}")

    with open(os.path.join(outputs_dir, "training_history.json"), "w") as f:
        json.dump(histories, f, indent=2, default=float)

    plot_history(histories, outputs_dir)
    print(f"Training plots saved to {outputs_dir}/")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train driver distraction CNN")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)
