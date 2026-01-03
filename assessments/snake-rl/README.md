# Deep Q-Learning Snake Agent

**MSc Artificial Intelligence - Final Assessment**

A reinforcement learning agent that learns to play Snake using Deep Q-Learning with experience replay.

## Overview

This project implements a DQN (Deep Q-Network) agent that learns optimal gameplay strategies through trial and error. The agent receives no explicit rules about how to play - it discovers winning strategies by maximising cumulative rewards.

## Architecture

```
State (11 features) → Dense(256, ReLU) → Dense(3) → Q-values
```

**State Representation (11 inputs):**
- Danger detection: straight, left, right (3 features)
- Current direction: up, down, left, right (4 features)
- Food location: relative to head (4 features)

**Actions (3 outputs):**
- Continue straight
- Turn left
- Turn right

## Key Components

| File | Purpose |
|------|---------|
| `agent.py` | DQN agent with epsilon-greedy action selection |
| `model.py` | PyTorch neural network and Q-trainer |
| `game.py` | Snake game environment |
| `helper.py` | Plotting utilities |
| `run_experiment.py` | Hyperparameter experiment runner |
| `generate_plots.py` | Results visualisation |

## Algorithm

1. **Observe** current state (11 features)
2. **Select action** using epsilon-greedy (explore vs exploit)
3. **Execute** action, receive reward
4. **Store** experience in replay buffer
5. **Sample** batch from buffer
6. **Update** Q-values using Bellman equation:
   ```
   Q(s,a) = r + γ * max(Q(s',a'))
   ```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.001 |
| Gamma (discount) | 0.9 |
| Epsilon Start | 80 |
| Epsilon Decay | -1 per game |
| Replay Buffer | 100,000 |
| Batch Size | 1,000 |

## Results

- **Best Score:** 40+ consistently after 200 episodes
- **Key Finding:** Wider networks (256 neurons) outperformed deeper architectures
- **Learning Curve:** Initial random exploration → Basic survival → Food-seeking → Strategy optimisation

## Running

```bash
# Train agent
python agent.py

# Run experiments
python run_experiment.py

# Generate plots
python generate_plots.py
```

## Dependencies

```bash
pip install torch pygame numpy matplotlib
```

## References

- Mnih et al. (2013) - "Playing Atari with Deep Reinforcement Learning"
- Bellman Equation - Foundation of Q-Learning
