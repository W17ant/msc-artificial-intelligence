#!/usr/bin/env python3
# ###########################################################
#   ###   2418521        Antony O'Neill                   ###
#   ###   SNAKE RL - EXPERIMENT RUNNER                    ###
#   ###   Last Updated: 08-12-2025                        ###
# ###########################################################

"""
Run a specific experiment configuration and save results.
Usage: python run_experiment.py
"""

import os
os.environ['SDL_VIDEO_MAC_FULLSCREEN_SPACES'] = '0'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['SDL_MAC_DISABLE_SAVE_STATE'] = '1'

import torch
import random
import numpy as np
import pandas as pd
import time
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer


# ===============================================
# EXPERIMENT CONFIGURATION - MODIFY HERE
# ===============================================
EXPERIMENT_NAME = 'nn_wide_wall'
HIDDEN_LAYERS = [512]  # Wide network
HAS_WALL = True
MEMORY_SIZE = 100_000
TOTAL_EPISODES = 500
# ===============================================

BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
RESULTS_DIR = './experiment_results'


class ExperimentAgent:
    def __init__(self, hidden_layers, memory_size):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=memory_size)

        # Create network with specified architecture
        self.model = Linear_QNet(11, hidden_layers, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
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
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def save_results(scores, mean_scores, record, start_time, hidden_layers, memory_size, has_wall):
    """Save experiment results to CSV files"""
    exp_dir = os.path.join(RESULTS_DIR, EXPERIMENT_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    # Save scores.csv
    scores_df = pd.DataFrame({
        'Episode': range(1, len(scores) + 1),
        'Score': scores,
        'Mean_Score': [round(m, 2) for m in mean_scores]
    })
    scores_df.to_csv(os.path.join(exp_dir, 'scores.csv'), index=False)

    # Save averages.csv (per 100 episodes)
    averages_data = []
    for i in range(0, len(scores), 100):
        chunk = scores[i:i+100]
        if chunk:
            averages_data.append({
                'Episode_Range': f'{i+1}-{i+len(chunk)}',
                'Average': round(np.mean(chunk), 2),
                'Max': max(chunk),
                'Min': min(chunk)
            })
    averages_df = pd.DataFrame(averages_data)
    averages_df.to_csv(os.path.join(exp_dir, 'averages.csv'), index=False)

    # Calculate avg last 100
    avg_last_100 = round(np.mean(scores[-100:]), 2) if len(scores) >= 100 else round(np.mean(scores), 2)

    # Save summary.csv
    elapsed_time = time.time() - start_time
    hidden_str = str(hidden_layers) if isinstance(hidden_layers, list) else f'[{hidden_layers}]'

    # Determine description
    if hidden_layers == [512]:
        desc = 'Wide 2-layer network (11 -> 512 -> 3) with wall'
    else:
        desc = f'Network {hidden_str}'

    summary_data = [
        ('Experiment', EXPERIMENT_NAME),
        ('Description', desc),
        ('Total Episodes', len(scores)),
        ('Final Record', record),
        ('Final Mean Score', round(mean_scores[-1], 2)),
        ('Avg Last 100 Episodes', avg_last_100),
        ('Total Time (seconds)', round(elapsed_time, 1)),
        ('Hidden Layers', hidden_str),
        ('Memory Size', memory_size),
        ('Has Wall', has_wall),
    ]
    summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
    summary_df.to_csv(os.path.join(exp_dir, 'summary.csv'), index=False)

    print(f"\nResults saved to {exp_dir}/")


def run_experiment():
    print(f"Running experiment: {EXPERIMENT_NAME}")
    print(f"  Hidden layers: {HIDDEN_LAYERS}")
    print(f"  Memory size: {MEMORY_SIZE}")
    print(f"  Wall: {HAS_WALL}")
    print(f"  Episodes: {TOTAL_EPISODES}")
    print("-" * 40)

    agent = ExperimentAgent(HIDDEN_LAYERS, MEMORY_SIZE)
    game = SnakeGameAI(has_wall=HAS_WALL)

    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    start_time = time.time()

    while agent.n_games < TOTAL_EPISODES:
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

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)

            print(f'Game {agent.n_games}/{TOTAL_EPISODES} | Score: {score} | Record: {record} | Mean: {mean_score:.2f}')

    # Save results
    save_results(scores, mean_scores, record, start_time, HIDDEN_LAYERS, MEMORY_SIZE, HAS_WALL)
    print(f"\nExperiment complete! Record: {record}, Avg Last 100: {np.mean(scores[-100:]):.2f}")


if __name__ == '__main__':
    run_experiment()
