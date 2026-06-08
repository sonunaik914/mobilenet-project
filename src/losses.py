"""Custom loss functions."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Custom")
class SparseCategoricalCrossentropyWithLabelSmoothing(tf.keras.losses.Loss):
    """Sparse categorical cross entropy with label smoothing.

    Keras has label smoothing for categorical labels, but CIFAR-10 labels are sparse
    integers. This custom loss converts sparse labels to one-hot labels, applies
    label smoothing, and then calculates categorical cross entropy.
    """

    def __init__(self, num_classes: int = 10, label_smoothing: float = 0.1, name: str = "custom_label_smoothing_loss"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=self.num_classes)

        smooth_positive = 1.0 - self.label_smoothing
        smooth_negative = self.label_smoothing / tf.cast(self.num_classes, tf.float32)
        y_true_smooth = y_true_one_hot * smooth_positive + smooth_negative

        return tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config
