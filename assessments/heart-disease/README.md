# Heart Disease Classification

**MSc Artificial Intelligence - Mid-Module Assessment**

Binary classification comparing Random Forest, SVM, and Neural Network approaches for predicting heart disease risk.

## Overview

This project uses the UCI Heart Disease dataset to predict whether a patient is at high risk of heart disease based on 13 clinical features. Three different machine learning approaches are compared, each with multiple configurations.

## Dataset

**Source:** UCI Heart Disease Dataset
**Samples:** 303 patients
**Features:** 13 clinical variables
**Target:** Binary (0 = Low risk, 1 = High risk)

### Features

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Biological sex (0=F, 1=M) |
| cp | Chest pain type (0-3) |
| trtbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results (0-2) |
| thalachh | Maximum heart rate achieved |
| exng | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slp | Slope of peak exercise ST segment |
| caa | Number of major vessels (0-4) |
| thall | Thalassemia (0-3) |

## Models Tested

### Random Forest (Ensemble)
| Config | Parameters | Accuracy |
|--------|------------|----------|
| 1 | Default (100 trees) | 83.61% |
| 2 | 200 trees, max_depth=10 | 81.97% |
| 3 | min_samples_split=5, min_samples_leaf=2 | **85.25%** |

### Support Vector Machine (Kernel)
| Config | Parameters | Accuracy |
|--------|------------|----------|
| 1 | RBF kernel | 81.97% |
| 2 | Linear kernel | 78.69% |
| 3 | RBF, C=10, gamma=0.1 | 75.41% |

### Neural Network (Deep Learning)
| Config | Architecture | Accuracy |
|--------|--------------|----------|
| 1 | 64-32 | 77.05% |
| 2 | 128-64-32 + Dropout | 77.05% |
| 3 | 256-128-64 + Dropout | 80.33% |

## Key Findings

1. **Best Model:** Random Forest with sample constraints (85.25%)
2. **Most Predictive Features:**
   - Chest pain type (cp)
   - Maximum heart rate (thalachh)
   - Number of major vessels (caa)
3. **Neural Networks:** Showed tendency to overfit on small dataset despite dropout

## Visualisations

The notebook includes:
- Target distribution analysis
- Feature correlation heatmap
- Box plots by target class
- ROC curves for all models
- Confusion matrices
- Feature importance ranking
- Model comparison bar chart

## Running

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

# Run notebook
jupyter notebook Heart_Disease_Classification.ipynb
```

## Files

| File | Description |
|------|-------------|
| `Heart_Disease_Classification.ipynb` | Complete analysis notebook |
| `heart.csv` | UCI Heart Disease dataset |

## References

- UCI Machine Learning Repository - Heart Disease Dataset
- scikit-learn documentation
- TensorFlow/Keras documentation
