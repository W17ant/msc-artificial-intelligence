# MSc Artificial Intelligence Projects

Machine learning and AI projects developed during my MSc Computer Science with Artificial Intelligence at St Mary's University.

## Projects

### 1. CNN Image Classification (CIFAR-10)
**File:** `CNN_CIFAR10_CLEAN.ipynb`

Convolutional Neural Network for classifying images from the CIFAR-10 dataset (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

**Architecture:**
- 3 Convolutional blocks with BatchNormalization
- MaxPooling layers
- Dense layers with Dropout regularization
- ~170K trainable parameters

**Results:**
- Test Accuracy: **71.33%**
- Training: 20 epochs with Adam optimizer (lr=0.0001)

**Technologies:** TensorFlow, Keras, NumPy, Matplotlib

---

### 2. Deep Q-Network (CartPole)
**File:** `DQN_CartPole.ipynb`

Reinforcement learning agent trained to balance a pole on a cart using Deep Q-Learning.

**Key Concepts:**
- Experience Replay with ReplayMemory buffer
- Target Network for stable learning
- Epsilon-greedy action selection with decay
- Soft updates (TAU=0.005)

**Hyperparameters:**
- Batch Size: 128
- Gamma (discount): 0.99
- Learning Rate: 1e-4
- Episodes: 600

**Results:**
- Max Reward: 283
- Average (last 100 episodes): ~109

**Technologies:** PyTorch, Gymnasium (OpenAI Gym), NumPy

---

### 3. Classification Fundamentals
**File:** `CLASSIFICATION.ipynb`

Foundational classification techniques and algorithms.

**Technologies:** Scikit-learn, Pandas, NumPy

---

### 4. AI Ethics Interactive Game
**File:** `ai-ethics-finance-scenario.html`

Interactive choose-your-own-adventure exploring ethical challenges in AI-powered credit scoring systems.

**Features:**
- Branching narrative with ethical dilemmas
- Framework scoring (Utilitarianism, Deontology, Virtue Ethics)
- Shareable ending cards

**Live Demo:** [Play the game](https://aoneill.co.uk/MSC/ai-ethics-finance-scenario.html)

---

## Setup

### Requirements
```bash
pip install tensorflow torch gymnasium scikit-learn pandas numpy matplotlib
```

### Running Notebooks
```bash
jupyter notebook
```

## Author

**Antony O'Neill**
- MSc Computer Science with AI, St Mary's University
- [Portfolio](https://aoneill.co.uk)
- [GitHub](https://github.com/W17ant)
- [LinkedIn](https://www.linkedin.com/in/antony-o-neill-96601a104/)

## License

This project is for educational purposes as part of my MSc coursework.
