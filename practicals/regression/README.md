# House Price Prediction - Neural Network Regression

Deep neural network for predicting house prices using the King County dataset.

## Overview

This practical implements a multi-layer neural network for regression, predicting house sale prices based on property features.

## Dataset

- **Source:** King County House Sales (Kaggle)
- **Samples:** ~21,000 homes
- **Features:** 18 (bedrooms, bathrooms, sqft, location, etc.)
- **Target:** Sale price (continuous)

## Architecture

```
Input (18 features) → Dense(30, ReLU) → Dense(30, ReLU) → Dense(40, ReLU) → Dense(40, ReLU) → Dense(40, ReLU) → Output(1)
```

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Mean Absolute Error (MAE) |
| Epochs | 150 |
| Batch Size | 32 |
| Train/Test Split | 80/20 |

## Preprocessing

- MinMaxScaler normalisation on features
- Dropped rows with missing values

## Running

```bash
jupyter notebook KC_House_Dataset.ipynb
```

## Files

- `KC_House_Dataset.ipynb` — Main analysis notebook
- `kc_house_data.csv` — King County dataset
