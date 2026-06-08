"""Train multiple MobileNet models and compare accuracy vs size."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def run_training(alpha: float, epochs: int, batch_size: int) -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "src" / "train.py"
    command = [
        sys.executable,
        str(train_script),
        "--model",
        "mobilenet",
        "--alpha",
        str(alpha),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
    ]
    subprocess.run(command, check=True, cwd=project_root)


def load_metrics(alpha: float) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    metrics_path = project_root / "artifacts" / f"metrics_mobilenet_alpha_{alpha:.2f}.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_accuracy_vs_size(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df["tflite_model_size_mb"], df["test_accuracy"], marker="o")

    for _, row in df.iterrows():
        label = f"alpha={row['alpha']:.2f}"
        plt.annotate(label, (row["tflite_model_size_mb"], row["test_accuracy"]), textcoords="offset points", xytext=(5, 5))

    plt.xlabel("TFLite Model Size (MB)")
    plt.ylabel("Test Accuracy")
    plt.title("MobileNet CIFAR-10: Accuracy vs Model Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def compare(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    rows = []
    for alpha in args.alphas:
        print(f"\nTraining MobileNet alpha={alpha}...")
        run_training(alpha, args.epochs, args.batch_size)
        rows.append(load_metrics(alpha))

    df = pd.DataFrame(rows)
    selected_columns = [
        "model",
        "alpha",
        "test_accuracy",
        "test_loss",
        "trainable_parameters",
        "keras_model_size_mb",
        "tflite_model_size_mb",
        "epochs_trained",
    ]
    df = df[selected_columns].sort_values("alpha")

    comparison_path = results_dir / "comparison.csv"
    plot_path = results_dir / "accuracy_vs_size.png"
    df.to_csv(comparison_path, index=False)
    plot_accuracy_vs_size(df, plot_path)

    print("\nComparison complete:")
    print(df)
    print(f"\nSaved: {comparison_path}")
    print(f"Saved: {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MobileNet sizes on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for each model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0], help="MobileNet alpha values")
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse_args())
