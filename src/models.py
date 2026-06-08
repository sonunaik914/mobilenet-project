"""Model definitions for MobileNet-style CIFAR-10 classification."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def _make_divisible(value: float, divisor: int = 8) -> int:
    """Round filter number so it is divisible by divisor."""
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv_bn_relu(x, filters: int, kernel_size: int = 3, stride: int = 1, name: str | None = None):
    """Standard Conv2D + BatchNorm + ReLU6 block."""
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same", use_bias=False, name=None if name is None else f"{name}_conv")(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = layers.ReLU(max_value=6.0, name=None if name is None else f"{name}_relu6")(x)
    return x


def depthwise_separable_block(x, pointwise_filters: int, stride: int = 1, block_id: int = 1):
    """Depthwise separable convolution block used by MobileNet."""
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"dw_{block_id}",
    )(x)
    x = layers.BatchNormalization(name=f"dw_{block_id}_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"dw_{block_id}_relu6")(x)

    x = layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"pw_{block_id}",
    )(x)
    x = layers.BatchNormalization(name=f"pw_{block_id}_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"pw_{block_id}_relu6")(x)
    return x


def build_mobilenet_cifar10(input_shape=(32, 32, 3), num_classes: int = 10, alpha: float = 0.5, dropout_rate: float = 0.2):
    """Build a MobileNetV1-style model adapted for 32x32 CIFAR-10 images.

    Args:
        input_shape: Input image shape.
        num_classes: Number of output classes.
        alpha: Width multiplier. Lower alpha creates a smaller model.
        dropout_rate: Dropout before final classifier.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    x = augmentation(inputs)

    first_filters = _make_divisible(32 * alpha)
    x = conv_bn_relu(x, first_filters, kernel_size=3, stride=1, name="initial")

    block_settings = [
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (1024, 2),
        (1024, 1),
    ]

    for block_id, (filters, stride) in enumerate(block_settings, start=1):
        pointwise_filters = _make_divisible(filters * alpha)
        x = depthwise_separable_block(x, pointwise_filters, stride=stride, block_id=block_id)

    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"MobileNet_CIFAR10_alpha_{alpha}")
    return model


def build_baseline_cnn(input_shape=(32, 32, 3), num_classes: int = 10, dropout_rate: float = 0.3):
    """Build a simple baseline CNN for comparison."""
    inputs = layers.Input(shape=input_shape, name="input_image")

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.1, 0.1),
        ],
        name="data_augmentation",
    )

    x = augmentation(inputs)

    x = conv_bn_relu(x, 32, 3, 1, name="base_conv1")
    x = conv_bn_relu(x, 32, 3, 1, name="base_conv2")
    x = layers.MaxPooling2D(name="pool1")(x)

    x = conv_bn_relu(x, 64, 3, 1, name="base_conv3")
    x = conv_bn_relu(x, 64, 3, 1, name="base_conv4")
    x = layers.MaxPooling2D(name="pool2")(x)

    x = conv_bn_relu(x, 128, 3, 1, name="base_conv5")
    x = conv_bn_relu(x, 128, 3, 1, name="base_conv6")
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="Baseline_CNN_CIFAR10")


def count_trainable_params(model: tf.keras.Model) -> int:
    """Return number of trainable parameters."""
    return int(sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights))


def get_model(model_name: str = "mobilenet", alpha: float = 0.5, num_classes: int = 10):
    """Factory function for models."""
    model_name = model_name.lower()
    if model_name == "mobilenet":
        return build_mobilenet_cifar10(num_classes=num_classes, alpha=alpha)
    if model_name == "baseline":
        return build_baseline_cnn(num_classes=num_classes)
    raise ValueError(f"Unknown model_name: {model_name}. Use 'mobilenet' or 'baseline'.")
