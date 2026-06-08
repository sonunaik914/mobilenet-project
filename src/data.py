"""Data loading and preprocessing for CIFAR-10."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_cifar10(validation_size: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 and create train/validation/test splits.

    Args:
        validation_size: Number of samples taken from training data for validation.

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test
    """
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_full = y_train_full.reshape(-1).astype("int32")
    y_test = y_test.reshape(-1).astype("int32")

    x_val = x_train_full[-validation_size:]
    y_val = y_train_full[-validation_size:]
    x_train = x_train_full[:-validation_size]
    y_train = y_train_full[:-validation_size]

    return x_train, y_train, x_val, y_val, x_test, y_test


def make_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, training: bool = False) -> tf.data.Dataset:
    """Create a TensorFlow dataset pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        dataset = dataset.shuffle(buffer_size=len(x), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
