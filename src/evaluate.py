"""Evaluate a saved CIFAR-10 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data import CLASS_NAMES, load_cifar10, make_dataset
from losses import SparseCategoricalCrossentropyWithLabelSmoothing


def evaluate(model_path: str, batch_size: int = 64) -> None:
    _, _, _, _, x_test, y_test = load_cifar10()
    test_ds = make_dataset(x_test, y_test, batch_size, training=False)

    custom_objects = {
        "SparseCategoricalCrossentropyWithLabelSmoothing": SparseCategoricalCrossentropyWithLabelSmoothing,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved model on CIFAR-10")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .keras model")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    evaluate(args.model_path, args.batch_size)
