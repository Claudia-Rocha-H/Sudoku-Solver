from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

from .cnn_architecture import build_digit_cnn
from .cnn_data import build_tf_datasets, load_mnist_digits_1_to_9

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_DEFAULT_MODEL_PATH = os.path.join(_MODEL_DIR, "digit_cnn.keras")


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    validation_split: float = 0.1
    learning_rate: float = 1e-3
    model_path: str = _DEFAULT_MODEL_PATH


def train_and_save_digit_cnn(config: TrainConfig | None = None, status_callback=None):
    config = config or TrainConfig()

    try:
        tf = importlib.import_module("tensorflow")
    except ImportError as exc:
        raise RuntimeError("TensorFlow is missing. Install it with: pip install tensorflow") from exc

    np = importlib.import_module("numpy")

    if status_callback:
        status_callback("Preparing MNIST dataset (1-9)...", "#2f4858")

    (x_train, y_train), (x_val, y_val) = load_mnist_digits_1_to_9(
        tf,
        np,
        validation_split=config.validation_split,
    )

    train_ds, val_ds = build_tf_datasets(
        tf,
        x_train,
        y_train,
        x_val,
        y_val,
        batch_size=config.batch_size,
    )

    if status_callback:
        status_callback("Training digit CNN model...", "#8a4b08")

    model = build_digit_cnn(tf, learning_rate=config.learning_rate)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=0,
    )

    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    model.save(config.model_path)

    if status_callback:
        best_val = max(history.history.get("val_accuracy", [0.0]))
        status_callback(f"CNN ready. val_accuracy={best_val:.4f}", "#2f4858")

    return {
        "model_path": config.model_path,
        "epochs_ran": len(history.history.get("loss", [])),
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
    }
