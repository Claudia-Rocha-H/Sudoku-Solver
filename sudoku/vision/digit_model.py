from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_TF_MODEL_PATH = os.path.join(_MODEL_DIR, "digit_cnn.keras")
_FEATURE_SIZE = 28


@dataclass
class Prediction:
    digit: int
    confidence: float
    margin: float = 0.0


class BaseDigitClassifier:
    def predict(self, digit_image_28x28):
        raise NotImplementedError

    def predict_many(self, digit_images_28x28):
        return [self.predict(image) for image in digit_images_28x28]


class TensorflowDigitClassifier(BaseDigitClassifier):
    def __init__(self, model, np_module):
        self.model = model
        self.np = np_module

    def predict(self, digit_image_28x28):
        x = (1.0 - digit_image_28x28.astype("float32") / 255.0).reshape(1, _FEATURE_SIZE, _FEATURE_SIZE, 1)
        probs = self.model(x, training=False).numpy()[0]
        order = self.np.argsort(probs)
        best_index = int(order[-1])
        second_index = int(order[-2]) if len(order) > 1 else best_index
        best_conf = float(probs[best_index])
        second_conf = float(probs[second_index])
        return Prediction(digit=best_index + 1, confidence=best_conf, margin=best_conf - second_conf)

    def predict_many(self, digit_images_28x28):
        if not digit_images_28x28:
            return []

        x = self.np.stack(
            [(1.0 - image.astype("float32") / 255.0) for image in digit_images_28x28],
            axis=0,
        )
        x = x.reshape((-1, _FEATURE_SIZE, _FEATURE_SIZE, 1))
        probs = self.model(x, training=False).numpy()

        predictions = []
        for row in probs:
            order = self.np.argsort(row)
            best_index = int(order[-1])
            second_index = int(order[-2]) if len(order) > 1 else best_index
            best_conf = float(row[best_index])
            second_conf = float(row[second_index])
            predictions.append(
                Prediction(
                    digit=best_index + 1,
                    confidence=best_conf,
                    margin=best_conf - second_conf,
                )
            )
        return predictions


def _ensure_tensorflow_available():
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        return importlib.import_module("tensorflow")
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is missing. Install it with: pip install tensorflow"
        ) from exc


def _get_tensorflow_classifier():
    if not os.path.exists(_TF_MODEL_PATH):
        return None

    np = importlib.import_module("numpy")
    _ensure_tensorflow_available()
    tf = importlib.import_module("tensorflow")
    model = tf.keras.models.load_model(_TF_MODEL_PATH)
    return TensorflowDigitClassifier(model, np)


def train_digit_classifier(status_callback=None):
    _ensure_tensorflow_available()
    from .cnn_trainer import TrainConfig, train_and_save_digit_cnn

    return train_and_save_digit_cnn(config=TrainConfig(model_path=_TF_MODEL_PATH), status_callback=status_callback)


def get_digit_classifier(status_callback=None, auto_train=False):
    try:
        tf_classifier = _get_tensorflow_classifier()
        if tf_classifier is not None:
            if status_callback:
                status_callback("Using local CNN model.", "#2f4858")
            return tf_classifier
    except Exception:
        if not auto_train:
            raise RuntimeError("Could not load the local CNN model.")

    if not auto_train:
        raise RuntimeError(
            "No CNN model found at sudoku/vision/models/digit_cnn.keras. "
            "Train it first with: python train_cnn.py"
        )

    train_digit_classifier(status_callback=status_callback)
    tf_classifier = _get_tensorflow_classifier()
    if tf_classifier is None:
        raise RuntimeError("Could not load the CNN model after training.")
    return tf_classifier
