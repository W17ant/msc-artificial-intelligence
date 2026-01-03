# ###########################################################
#   ###   2418521        Antony O'Neill                   ###
#   ###   SNAKE RL - AGENT                                ###
#   ###   Last Updated: 08-12-2025                        ###
# ###########################################################

# ===============================================
#   SNAKE RL - agent.py - DQN Agent
# ===============================================


# ===============================================
# 1. IMPORTS
# ===============================================
# External libraries for neural networks, data handling,
# and game environment interaction.

import os
os.environ['SDL_VIDEO_MAC_FULLSCREEN_SPACES'] = '0'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['SDL_HINT_MAC_CTRL_CLICK_EMULATE_RIGHT_CLICK'] = '1'
os.environ['SDL_MAC_DISABLE_SAVE_STATE'] = '1'

import torch
import random
import numpy as np
import json
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot


# ===============================================
# 2. CONFIGURATION
# ===============================================
# Hyperparameters for the DQN agent including memory size,
# batch size for training, and learning rate.

HISTORY_FILE = './model/training_history.json'
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


# ===============================================
# 3. AGENT CLASS
# ===============================================
# DQN Agent that learns to play Snake using Q-learning.
# Handles state extraction, action selection (epsilon-greedy),
# and experience replay for training.
#
# Changes made to original:
# - Added model.load() call in __init__ to resume from saved weights

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Exploration vs exploitation trade-off
        self.gamma = 0.9  # Discount rate: higher = more focus on future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # Auto-removes old memory when full

        # Neural network: 11 input states, 256 hidden neurons, 3 output actions
        self.model = Linear_QNet(11, 256, 3)
        self.model.load()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """Get the current state of the game as a feature vector"""
        head = game.snake[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Train on a batch of experiences (experience replay)"""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train on a single step"""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Get action based on current state (exploration vs exploitation)"""
        self.epsilon = 80 - self.n_games

        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            # Exploration: random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation: use model prediction
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


# ===============================================
# 4. HISTORY FUNCTIONS
# ===============================================
# Functions to save and load training progress,
# allowing training to resume from previous sessions.
#
# Changes made to original:
# - NEW SECTION: Added HISTORY_FILE constant for training history persistence
# - Added load_history() function to restore scores, mean_scores, total_score, record from JSON
# - Added save_history() function to persist training progress after each game

def load_history():
    """Load training history from file"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)
            print(f"Loaded training history: {len(data['scores'])} games, record: {data['record']}")
            return data
    return None


def save_history(scores, mean_scores, total_score, record):
    """Save training history to file"""
    data = {
        'scores': scores,
        'mean_scores': mean_scores,
        'total_score': total_score,
        'record': record
    }
    with open(HISTORY_FILE, 'w') as f:
        json.dump(data, f)


# ===============================================
# 5. TRAINING LOOP
# ===============================================
# Main training loop that runs episodes, collects experiences,
# and trains the neural network using the DQN algorithm.
#
# Changes made to original:
# - Training loop now loads history at start to resume from previous sessions
# - Agent's n_games counter restored from history length
# - save_history() called after each game to persist progress

def train():
    """Main training loop"""
    history = load_history()
    if history:
        plot_scores = history['scores']
        plot_mean_scores = history['mean_scores']
        total_score = history['total_score']
        record = history['record']
    else:
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0

    agent = Agent()

    if history:
        agent.n_games = len(plot_scores)

    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games} | Score: {score} | Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            save_history(plot_scores, plot_mean_scores, total_score, record)


# ===============================================
# 6. MAIN
# ===============================================
# Entry point - starts the training process.

if __name__ == '__main__':
    train()
