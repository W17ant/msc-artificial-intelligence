# ###########################################################
#   ###   2418521        Antony O'Neill                   ###
#   ###   SNAKE RL - GAME ENVIRONMENT                     ###
#   ###   Last Updated: 08-12-2025                        ###
# ###########################################################

# ===============================================
#   SNAKE RL - game.py - Game Environment
# ===============================================


# ===============================================
# 1. IMPORTS
# ===============================================
# External libraries for game rendering, random food placement, directional movement,
# and numerical operations for the AI agent's state representation.

import os
os.environ['SDL_VIDEO_MAC_FULLSCREEN_SPACES'] = '0'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['SDL_MAC_DISABLE_SAVE_STATE'] = '1'

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np


# ===============================================
# 2. CONFIGURATION
# ===============================================
# Game initialization, constants definition (colors, block size, speed),
# and setup of direction enums and point data structures.

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# RGB Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

BLOCK_SIZE = 20
SPEED = 40


# ===============================================
# 3. GAME CLASS
# ===============================================
# Main game environment class implementing the Snake game logic with AI agent interface,
# including collision detection, movement mechanics, reward system, and rendering.
#
# Changes made to original:
# - Added 'has_wall' parameter to enable/disable wall obstacles
# - Added _create_walls() method to generate static barriers in center of game
# - Updated collision detection to check for wall collisions
# - Updated _place_food() to avoid spawning food on walls
# - Updated _update_ui() to render wall obstacles in gray
# - Snake spawns in top-left quadrant when walls are enabled to avoid immediate collision

class SnakeGameAI:
    def __init__(self, w=640, h=480, has_wall=False):
        self.w = w
        self.h = h
        self.has_wall = has_wall

        self.display = pygame.display.set_mode((self.w, self.h))
        caption = 'Snake RL' + (' (with Wall)' if has_wall else '')
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()

        self.walls = []
        if has_wall:
            self._create_walls()

        self.reset()

    def _create_walls(self):
        """Create static wall obstacles in the center of the game"""
        center_x = self.w // 2
        center_y = self.h // 2

        for i in range(-2, 3):
            self.walls.append(Point(center_x + i * BLOCK_SIZE, center_y))

        for i in range(-1, 2):
            if i != 0:
                self.walls.append(Point(center_x, center_y + i * BLOCK_SIZE))

    def reset(self):
        """Reset the game state"""
        self.direction = Direction.RIGHT
        if self.has_wall:
            self.head = Point(self.w // 4, self.h // 4)
        else:
            self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        """Place food at random position"""
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.walls:
            self._place_food()

    def play_step(self, action):
        """Execute one step of the game with given action from agent"""
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False

        # Check collision OR if snake is stuck in loop
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10  # Penalty for dying
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10  # Reward for eating food
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """Check if there is a collision"""
        if pt is None:
            pt = self.head

        if (pt.x > self.w - BLOCK_SIZE or pt.x < 0 or
            pt.y > self.h - BLOCK_SIZE or pt.y < 0):
            return True

        if pt in self.snake[1:]:
            return True

        if pt in self.walls:
            return True

        return False

    def _update_ui(self):
        """Update the game display"""
        self.display.fill(BLACK)

        for wall in self.walls:
            pygame.draw.rect(self.display, GRAY, pygame.Rect(wall.x, wall.y, BLOCK_SIZE, BLOCK_SIZE))

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

        pygame.display.flip()

    def _move(self, action):
        """Move based on action: [straight, right turn, left turn]"""
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # No change - go straight
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4  # Right turn
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4  # Left turn
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
