# CartPole DQN - Reinforcement Learning

Deep Q-Network agent learning to balance a pole on a cart.

## Overview

This practical implements a DQN agent for the classic CartPole control problem using PyTorch and Gymnasium.

## Environment

- **Task:** Balance a pole attached to a cart by moving left/right
- **State:** 4 values (cart position, velocity, pole angle, angular velocity)
- **Actions:** 2 (push left, push right)
- **Reward:** +1 for each timestep the pole remains upright
- **Termination:** Pole angle > 12°, cart position > 2.4, or 500 steps

## Architecture

```
State (4) → Dense(128, ReLU) → Dense(128, ReLU) → Q-values (2)
```

## Key Techniques

| Technique | Purpose |
|-----------|---------|
| Experience Replay | Breaks correlation between consecutive samples |
| Target Network | Stabilises learning with delayed updates |
| Epsilon-Greedy | Balances exploration vs exploitation |
| Soft Updates | Gradual target network updates (τ=0.005) |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 128 |
| Gamma (discount) | 0.99 |
| Learning Rate | 1e-4 |
| Epsilon Start | 0.9 |
| Epsilon End | 0.05 |
| Epsilon Decay | 1000 |
| TAU (soft update) | 0.005 |
| Replay Buffer | 10,000 |

## Results

- **Episodes:** 600
- **Max Reward:** 283
- **Average (last 100):** ~109

## Running

```bash
jupyter notebook DQN_CartPole.ipynb
```

## Files

- `DQN_CartPole.ipynb` — Full implementation with training loop
