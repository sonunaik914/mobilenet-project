"""Train MobileNet on CIFAR-10."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from data import load_cifar10, make_dataset
from losses import SparseCategoricalCrossentropyWithLabelSmoothing
from models import count_trainable_params, get_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def export_tflite(model: tf.keras.Model, output_path: Path) -> bool:
    """Export model to TensorFlow Lite.

    Returns True if export succeeds, False otherwise.
    """
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"TFLite conversion failed: {exc}")
        return False


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"
    results_dir = project_root / "results"
    artifacts_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    x_train, y_train, x_val, y_val, x_test, y_test = load_cifar10(validation_size=args.validation_size)

    train_ds = make_dataset(x_train, y_train, args.batch_size, training=True)
    val_ds = make_dataset(x_val, y_val, args.batch_size, training=False)
    test_ds = make_dataset(x_test, y_test, args.batch_size, training=False)

    model = get_model(args.model, alpha=args.alpha, num_classes=10)

    loss = SparseCategoricalCrossentropyWithLabelSmoothing(
        num_classes=10,
        label_smoothing=args.label_smoothing,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )

    model_tag = f"{args.model}_alpha_{args.alpha:.2f}" if args.model == "mobilenet" else args.model
    model_path = artifacts_dir / f"{model_tag}.keras"
    tflite_path = artifacts_dir / f"{model_tag}.tflite"
    history_path = results_dir / f"history_{model_tag}.csv"
    metrics_path = artifacts_dir / f"metrics_{model_tag}.json"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(history_path)),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(model.summary())

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final/restored model.
    model.save(model_path)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    tflite_success = export_tflite(model, tflite_path)

    metrics = {
        "model": args.model,
        "alpha": args.alpha if args.model == "mobilenet" else None,
        "epochs_requested": args.epochs,
        "epochs_trained": len(history.history.get("loss", [])),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "label_smoothing": args.label_smoothing,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "trainable_parameters": count_trainable_params(model),
        "keras_model_size_mb": file_size_mb(model_path),
        "tflite_export_success": tflite_success,
        "tflite_model_size_mb": file_size_mb(tflite_path),
        "model_path": str(model_path),
        "tflite_path": str(tflite_path),
        "history_path": str(history_path),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\nTraining complete. Metrics:")
    print(json.dumps(metrics, indent=4))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNet on CIFAR-10")
    parser.add_argument("--model", type=str, default="mobilenet", choices=["mobilenet", "baseline"], help="Model type")
    parser.add_argument("--alpha", type=float, default=0.5, help="MobileNet width multiplier")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing value")
    parser.add_argument("--validation-size", type=int, default=5000, help="Validation samples from training split")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    train(parse_args())
