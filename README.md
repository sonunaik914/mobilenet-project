# MobileNet on CIFAR-10 for Lightweight Image Classification

This project trains a **MobileNet-style lightweight CNN** on the **CIFAR-10** image classification dataset and compares **accuracy vs model size** for edge-device use.

## 1. Project Objective

The goal is to build a compact image classification model that can run on limited-resource devices such as mobile phones, Raspberry Pi, or other edge devices.

We compare different MobileNet width multipliers:

- `alpha = 0.25` → smallest model, lower computation
- `alpha = 0.50` → balanced model
- `alpha = 1.00` → larger MobileNet model, usually higher accuracy

The comparison focuses on:

- Test accuracy
- Number of parameters
- Saved model size
- TensorFlow Lite model size

## 2. Dataset

Dataset: CIFAR-10  
Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

CIFAR-10 contains:

- 50,000 training images
- 10,000 test images
- Image size: 32 × 32 × 3
- 10 classes

The dataset is automatically downloaded using:

```python
tf.keras.datasets.cifar10.load_data()
```

## 3. Methodology

### Step 1: Load CIFAR-10

The dataset is loaded using Keras. Pixel values are normalized from `[0, 255]` to `[0, 1]`.

### Step 2: Data Augmentation

To reduce overfitting, the training images are augmented using:

- Random horizontal flip
- Random translation
- Random rotation
- Random zoom

### Step 3: Build MobileNet-style model

MobileNet is lightweight because it uses **depthwise separable convolution**.

A normal convolution applies filters across all input channels at once. Depthwise separable convolution splits this into two cheaper operations:

1. **Depthwise convolution**: applies one spatial filter per input channel.
2. **Pointwise convolution**: uses a 1×1 convolution to combine channels.

This reduces parameters and computation.

### Step 4: Train models with different sizes

The model is trained for different width multipliers:

```bash
alpha = 0.25
alpha = 0.50
alpha = 1.00
```

A smaller alpha means fewer filters, fewer parameters, and smaller model size.

### Step 5: Evaluate

After training, the project records:

- Training accuracy
- Validation accuracy
- Test accuracy
- Parameter count
- Keras model size
- TFLite model size

### Step 6: Compare accuracy vs size

The script `src/compare.py` trains multiple MobileNet versions and creates:

- `results/comparison.csv`
- `results/accuracy_vs_size.png`

## 4. Custom Loss Function

This project uses a custom loss:

```text
Sparse Categorical Cross Entropy with Label Smoothing
```

Normal cross entropy makes the model too confident. Label smoothing softens the target labels.

Example for 10 classes:

Normal one-hot label for class 3:

```text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

After label smoothing:

```text
[0.01, 0.01, 0.01, 0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
```

This helps the model generalize better and can reduce overfitting.

The implementation is in:

```text
src/losses.py
```

## 5. Project Structure

```text
mobilenet-cifar10-lightweight/
│
├── README.md
├── PROJECT_REPORT.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data.py
│   ├── losses.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare.py
│   └── predict.py
│
├── artifacts/
│   └── saved models are stored here
│
└── results/
    └── graphs, metrics, and reports are stored here
```

## 6. Installation

### Recommended Python version

Use Python 3.10, 3.11, or 3.12 for TensorFlow compatibility.

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 7. How to Run

### Train one MobileNet model

```bash
python src/train.py --model mobilenet --alpha 0.50 --epochs 20 --batch-size 64
```

### Train smaller model

```bash
python src/train.py --model mobilenet --alpha 0.25 --epochs 20 --batch-size 64
```

### Train larger model

```bash
python src/train.py --model mobilenet --alpha 1.00 --epochs 20 --batch-size 64
```

### Compare multiple model sizes

```bash
python src/compare.py --epochs 20 --batch-size 64 --alphas 0.25 0.50 1.00
```

This produces:

```text
results/comparison.csv
results/accuracy_vs_size.png
```

### Evaluate a saved model

```bash
python src/evaluate.py --model-path artifacts/mobilenet_alpha_0.50.keras
```

### Predict one image

```bash
python src/predict.py --model-path artifacts/mobilenet_alpha_0.50.keras --image-path path/to/image.png
```

## 8. Expected Output Files

After training, you will see files like:

```text
artifacts/mobilenet_alpha_0.50.keras
artifacts/mobilenet_alpha_0.50.tflite
artifacts/metrics_mobilenet_alpha_0.50.json
results/history_mobilenet_alpha_0.50.csv
results/comparison.csv
results/accuracy_vs_size.png
```

## 9. GitHub Upload Steps

```bash
git init
git add .
git commit -m "Initial MobileNet CIFAR-10 project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mobilenet-cifar10-lightweight.git
git push -u origin main
```

## 10. Conclusion

This project shows how MobileNet can be used for lightweight image classification. By changing the width multiplier, we can choose a suitable trade-off between accuracy and model size for edge devices.
