# Real vs AI Image Detector (CIFAKE)

A deep learning project for detecting AI-generated images vs real images using PyTorch and ResNet50 architecture trained on the CIFAKE dataset.

## 📋 Project Overview

This project implements a binary image classifier that distinguishes between real and AI-generated (fake) images. It includes training scripts, evaluation tools, and a user-friendly GUI for real-time image classification.

## 🚀 Features

- **ResNet50-based CNN** for image classification
- **Interactive GUI** for single image prediction
- **Comprehensive evaluation** with metrics and confusion matrix
- **Visualization tools** for results analysis
- **Batch prediction** capabilities
- **Model checkpointing** to save best performing models

## 📁 Project Structure

```
├── main_cifake.py           # Main training script
├── predict_single.py        # Single image prediction
├── gui_cifake.py           # GUI application for predictions
├── evaluate_test.py        # Evaluation on test set
├── visualize_results.py    # Visualization of results
├── requirements.txt        # Project dependencies
├── data/
│   └── CIFAKE/
│       ├── train/          # Training data (FAKE/REAL)
│       └── test/           # Test data (FAKE/REAL)
├── runs/
│   └── cifake_resnet50/
│       ├── best_model.pth  # Trained model weights
│       └── *.csv           # Prediction results
└── test_images/            # Sample images for testing
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "Detectarea imaginilor reale si AI"
```

> Repository folder name is in Romanian as required for academic submission.

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The project uses the **CIFAKE dataset**, which contains:
- Real images collected from CIFAR-10
- AI-generated synthetic images

📌 **Dataset source (Kaggle)**:  
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

⚠️ **Note**: The CIFAKE dataset is not included in this repository and must be downloaded separately due to size constraints.

Dataset structure:


```
data/CIFAKE/
├── train/
│   ├── FAKE/
│   └── REAL/
└── test/
    ├── FAKE/
    └── REAL/
```

## 🎯 Usage

### Training the Model

```bash
python main_cifake.py
```

This will:
- Train a ResNet50 model on the CIFAKE dataset
- Save the best model to `runs/cifake_resnet50/best_model.pth`
- Generate training logs and metrics

### GUI Application

```bash
python gui_cifake.py
```

Features:
- Load and classify images
- Visual prediction result (REAL / FAKE)
- Confidence score display
- Real-time prediction with confidence scores
- Dark-themed modern interface

### Single Image Prediction

```bash
python predict_single.py <path_to_image>
```

### Evaluate Model

```bash
python evaluate_test.py
```

Generates:
- Accuracy metrics
- Confusion matrix
- Classification report
- Prediction CSV files (correct/wrong predictions)

### Visualize Results

```bash
python visualize_results.py
```

## 🔧 Configuration

Key parameters in `main_cifake.py`:

- `IMG_SIZE`: 224 (input image size)
- `BATCH_SIZE`: 32
- `EPOCHS`: 5
- `LR`: 1e-3 (learning rate)
- `WEIGHT_DECAY`: 1e-4

## 📈 Model Architecture

- **Base**: ResNet50 (pretrained on ImageNet)
- **Modification**: Final fully connected layer adapted for binary classification
- **Optimizer**: Adam
- **Loss Function**: Cross Entropy Loss

## 🎨 Data Augmentation

Training augmentations:
- Random horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- Normalization (ImageNet statistics)

## 📊 Results

Model performance metrics are saved in:
- `runs/cifake_resnet50/test_predictions.csv`
- `runs/cifake_resnet50/correct_predictions.csv`
- `runs/cifake_resnet50/wrong_predictions.csv`

⚠️ **Note**: Trained model weights and datasets are not included in this repository due to size constraints. Please train the model locally to generate them.


