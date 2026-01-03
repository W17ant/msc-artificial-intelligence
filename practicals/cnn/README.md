# CIFAR-10 Image Classification - CNN

Convolutional Neural Network for classifying 10 categories of images.

## Overview

This practical implements a CNN to classify small colour images into 10 categories using the CIFAR-10 benchmark dataset.

## Dataset

- **Source:** CIFAR-10 (via TensorFlow/Keras)
- **Training:** 50,000 images
- **Testing:** 10,000 images
- **Image Size:** 32x32 RGB
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Architecture

```
Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.2)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.4)
    ↓
Flatten → Dense(128) → BatchNorm → Dropout(0.5) → Dense(10, Softmax)
```

**Total Parameters:** ~170,000

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.0001) |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 20 |
| Batch Size | 32 |

## Results

- **Test Accuracy:** 71.33%
- **Key Techniques:** BatchNormalization, Dropout regularization, progressive filter increase

## Running

```bash
jupyter notebook CNN_CIFAR10_CLEAN.ipynb
```

## Files

- `CNN_CIFAR10_CLEAN.ipynb` — Full training notebook with visualisations
