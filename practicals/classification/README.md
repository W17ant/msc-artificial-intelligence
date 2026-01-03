# Bank Note Authentication - Classification

Comparing 6 classification algorithms on the Bank Note Authentication dataset.

## Overview

This practical compares multiple classification algorithms to detect counterfeit bank notes based on image-derived features.

## Dataset

- **Source:** UCI Bank Note Authentication
- **Features:** 4 (variance, skewness, kurtosis, entropy of wavelet-transformed images)
- **Target:** Binary (0 = Genuine, 1 = Counterfeit)

## Algorithms Tested

| Algorithm | Accuracy | Notes |
|-----------|----------|-------|
| Logistic Regression | 99% | Fast, interpretable |
| **SVM** | **100%** | Best performance |
| Decision Tree | 99% | Prone to overfitting |
| Random Forest | 99% | Ensemble method |
| **KNN** | **100%** | Distance-based |
| Naive Bayes | 86% | Assumes feature independence |

## Key Findings

- SVM and KNN achieved perfect classification (100%)
- Naive Bayes performed worst due to violated independence assumptions
- Dataset is well-separated, making it suitable for linear classifiers

## Running

```bash
jupyter notebook CLASSIFICATION.ipynb
```

## Files

- `CLASSIFICATION.ipynb` — Main analysis notebook
- `BankNote_Authentication.csv` — Dataset
