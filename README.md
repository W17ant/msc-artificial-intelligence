# MSc Artificial Intelligence - Coursework Portfolio

**Antony O'Neill** | St Mary's University, Twickenham | 2024-2025

A collection of machine learning projects, practicals, and assessments from my MSc Computer Science with Artificial Intelligence degree.

---

## Repository Structure

```
msc-artificial-intelligence/
├── assessments/          # Graded coursework
│   ├── snake-rl/        # Deep Q-Learning Snake Agent (Final Assessment)
│   └── heart-disease/   # Heart Disease Classification (Mid-Module)
├── practicals/          # Hands-on learning exercises
│   ├── classification/  # Bank Note Authentication
│   ├── regression/      # House Price Prediction
│   ├── neural-networks/ # MNIST Handwriting Recognition
│   ├── cnn/            # CIFAR-10 Image Classification
│   └── reinforcement-learning/ # CartPole DQN
└── ethics/             # AI Ethics Interactive Game
```

---

## Assessments

### [Deep Q-Learning Snake Agent](./assessments/snake-rl/)
**Final Assessment** | Python, PyTorch, Reinforcement Learning

Reinforcement learning agent that learns to play Snake using a deep neural network with experience replay and epsilon-greedy exploration.

- **Architecture:** 11 inputs → 256 hidden → 3 outputs (straight, left, right)
- **Key Features:** Experience replay buffer (100K), Bellman equation updates, epsilon decay
- **Experiments:** 12 configurations testing architecture depth, hidden layer width, memory sizes
- **Results:** Consistent scores of 40+ after 200 training episodes

### [Heart Disease Classification](./assessments/heart-disease/)
**Mid-Module Assessment** | Python, scikit-learn, TensorFlow

Binary classification comparing multiple ML approaches for predicting heart disease risk using the UCI Heart Disease dataset.

- **Models Tested:** Random Forest (3 configs), SVM (3 configs), Neural Network (3 configs)
- **Features:** 13 clinical variables including age, cholesterol, blood pressure, ECG results
- **Best Result:** 85.25% accuracy with tuned Random Forest
- **Analysis:** ROC curves, confusion matrices, feature importance ranking

---

## Practicals

Hands-on exercises covering core machine learning concepts:

| Practical | Technique | Dataset | Key Learning |
|-----------|-----------|---------|--------------|
| [Classification](./practicals/classification/) | Logistic Regression, SVM, Decision Tree, Random Forest, KNN, Naive Bayes | Bank Note Authentication | Comparing 6 classifiers (SVM achieved 100%) |
| [Regression](./practicals/regression/) | Deep Neural Network (5 layers) | KC House Prices (21K homes) | Neural network regression with MAE loss |
| [Neural Networks](./practicals/neural-networks/) | Dense NN, CNN | MNIST (60K digits) | Image classification fundamentals |
| [CNN](./practicals/cnn/) | Convolutional Neural Network | CIFAR-10 (10 classes) | 71.33% accuracy with BatchNorm & Dropout |
| [Reinforcement Learning](./practicals/reinforcement-learning/) | Deep Q-Network | CartPole | Experience replay, target networks, epsilon-greedy |

---

## Ethics

### [The Credit Algorithm](./ethics/ai-ethics-finance-scenario.html)
Interactive fiction game exploring ethical challenges in AI-powered credit scoring systems.

- **Format:** Choose-your-own-adventure with branching narratives
- **Themes:** Algorithmic bias, transparency, fairness, accountability
- **Features:** Framework scoring (Utilitarianism, Deontology, Virtue Ethics), shareable endings
- **[Play Online](https://aoneill.co.uk/MSC/ai-ethics-finance-scenario.html)**

---

## Tech Stack

**Languages & Frameworks**
- Python, PyTorch, TensorFlow, Keras, scikit-learn

**Data & Visualisation**
- pandas, NumPy, Matplotlib, Seaborn

**Tools**
- Jupyter Notebooks, Git

---

## Setup

```bash
# Clone repository
git clone https://github.com/W17ant/msc-artificial-intelligence.git
cd msc-artificial-intelligence

# Install dependencies
pip install torch tensorflow scikit-learn pandas numpy matplotlib seaborn jupyter

# Run notebooks
jupyter notebook
```

---

## About

This repository documents my progression through the AI module of my MSc:

1. **Classification** - Understanding fundamental ML algorithms
2. **Regression** - Neural networks for continuous prediction
3. **Computer Vision** - CNNs for image classification
4. **Reinforcement Learning** - Agents that learn from interaction

For my full portfolio including web development and production projects, visit **[aoneill.co.uk](https://aoneill.co.uk)**

---

## Author

**Antony O'Neill**
- [Portfolio](https://aoneill.co.uk)
- [GitHub](https://github.com/W17ant)
- [LinkedIn](https://www.linkedin.com/in/AntonyONeillADL)

---

*MSc Computer Science with Artificial Intelligence - St Mary's University, Twickenham*
