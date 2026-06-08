# Project Report: MobileNet on CIFAR-10 for Lightweight Image Classification

## Abstract

This project implements a MobileNet-style lightweight convolutional neural network for CIFAR-10 image classification. The main purpose is to study the trade-off between classification accuracy and model size, which is important for deployment on edge devices.

## Problem Statement

Large deep learning models often give high accuracy but require high memory and computation. Edge devices have limited memory, low processing power, and limited battery. Therefore, a lightweight model is required.

## Dataset Description

CIFAR-10 is an image classification dataset containing 60,000 color images of size 32×32. It has 10 object classes:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The dataset is split into 50,000 training images and 10,000 test images.

## Proposed Method

The project uses a MobileNet-style architecture based on depthwise separable convolutions.

### Why MobileNet?

MobileNet is designed for mobile and embedded vision applications. Instead of using normal convolution everywhere, it uses depthwise separable convolution, which reduces the number of parameters and computations.

### Width Multiplier

The width multiplier `alpha` controls how many filters are used in the network.

- Lower alpha: smaller model, faster inference, possibly lower accuracy
- Higher alpha: larger model, slower inference, possibly higher accuracy

This project trains and compares:

- MobileNet alpha 0.25
- MobileNet alpha 0.50
- MobileNet alpha 1.00

## Custom Loss Function

The custom loss function used in this project is:

```text
Sparse Categorical Cross Entropy with Label Smoothing
```

### Formula idea

For a correct class label, normal one-hot encoding gives probability 1 to the correct class and 0 to all other classes. Label smoothing gives slightly less than 1 to the correct class and small values to the other classes.

This prevents the model from becoming overconfident.

### Why this loss is useful

- Reduces overfitting
- Improves generalization
- Helps when some dataset labels are noisy
- Makes the model less overconfident

## Training Details

Recommended training configuration:

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 20 |
| Loss | Custom label smoothing cross entropy |
| Metric | Accuracy |

## Evaluation Metrics

The project evaluates each model using:

- Test accuracy
- Number of trainable parameters
- Keras model size in MB
- TensorFlow Lite model size in MB

## Result Table Template

After running the project, fill this table using `results/comparison.csv`.

| Model | Alpha | Test Accuracy | Parameters | Keras Size MB | TFLite Size MB |
|---|---:|---:|---:|---:|---:|
| MobileNet | 0.25 | generated after training | generated | generated | generated |
| MobileNet | 0.50 | generated after training | generated | generated | generated |
| MobileNet | 1.00 | generated after training | generated | generated | generated |

## Conclusion

MobileNet is suitable for lightweight image classification because it reduces model size using depthwise separable convolutions. The final choice of alpha depends on the edge-device requirement. If the device has very limited memory, alpha 0.25 is useful. If better accuracy is required and the device can support a larger model, alpha 0.50 or alpha 1.00 may be selected.
