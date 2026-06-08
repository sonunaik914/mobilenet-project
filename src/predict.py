"""Predict the CIFAR-10 class of a single image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from data import CLASS_NAMES
from losses import SparseCategoricalCrossentropyWithLabelSmoothing


def load_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((32, 32))
    array = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(array, axis=0)


def predict(model_path: str, image_path: str) -> None:
    custom_objects = {
        "SparseCategoricalCrossentropyWithLabelSmoothing": SparseCategoricalCrossentropyWithLabelSmoothing,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    image = load_image(image_path)

    probs = model.predict(image, verbose=0)[0]
    class_index = int(np.argmax(probs))
    class_name = CLASS_NAMES[class_index]
    confidence = float(probs[class_index])

    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence:.4f}")

    print("\nAll class probabilities:")
    for name, prob in zip(CLASS_NAMES, probs):
        print(f"{name:>10}: {prob:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one CIFAR-10 image")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .keras model")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not Path(args.image_path).exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    predict(args.model_path, args.image_path)
