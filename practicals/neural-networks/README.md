# MNIST Handwriting Recognition

Neural networks for classifying handwritten digits (0-9).

## Overview

This practical explores neural network architectures for image classification using the classic MNIST dataset.

## Dataset

- **Source:** MNIST (via TensorFlow/Keras)
- **Training:** 60,000 images
- **Testing:** 10,000 images
- **Image Size:** 28x28 grayscale
- **Classes:** 10 (digits 0-9)

## Notebooks

### 1. Handwriting_practical.ipynb
Basic dense neural network approach:
- Flatten input → Dense layers with ReLU → Softmax output
- Adam optimizer with sparse categorical crossentropy

### 2. CNN_image_training.ipynb
Convolutional neural network approach:
- Conv2D layers for feature extraction
- MaxPooling for downsampling
- Dense layers for classification

## Key Concepts

- **Normalisation:** Pixel values scaled to 0-1 range
- **Softmax:** Probability distribution over 10 classes
- **Sparse Categorical Crossentropy:** Loss function for integer labels

## Running

```bash
jupyter notebook Handwriting_practical.ipynb
# or
jupyter notebook CNN_image_training.ipynb
```

## Files

- `Handwriting_practical.ipynb` — Dense network approach
- `CNN_image_training.ipynb` — CNN approach
