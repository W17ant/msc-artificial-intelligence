# ###########################################################
#   ###   2418521        Antony O'Neill                   ###
#   ###   SNAKE RL - NEURAL NETWORK MODEL                 ###
#   ###   Last Updated: 08-12-2025                        ###
# ###########################################################

# ===============================================
#   SNAKE RL - model.py - DQN Neural Network
# ===============================================


# ===============================================
# 1. IMPORTS
# ===============================================
# PyTorch libraries for building and training neural networks, plus OS utilities for file management.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


# ===============================================
# 2. NEURAL NETWORK CLASS
# ===============================================
# Deep Q-Network (DQN) implementation with configurable hidden layers for learning optimal actions.
#
# Changes made to original:
# - Modified to support variable hidden layers (single int or list of ints)
# - Replaced fixed linear1/linear2 with dynamic nn.ModuleList for flexible architectures
# - Added 'architecture' attribute for logging network structure
# - Added load() method to restore previously saved model weights

class Linear_QNet(nn.Module):
    """Neural network for Q-learning - supports variable hidden layers"""

    def __init__(self, input_size, hidden_size, output_size):
        # input_size: number of input features (state size)
        # hidden_size: int or list of ints for hidden layer dimensions
        # output_size: number of actions (Q-values to output)
        super().__init__()
        # Handle single int or list of hidden layer sizes
        if isinstance(hidden_size, int):
            hidden_layers = [hidden_size]
        else:
            hidden_layers = hidden_size

        # Create dynamic list of network layers
        self.layers = nn.ModuleList()
        prev_size = input_size

        # Build hidden layers sequentially
        for h_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, h_size))
            prev_size = h_size

        # Output layer - raw Q values for each action
        self.output_layer = nn.Linear(prev_size, output_size)
        # Store architecture as readable string for logging
        self.architecture = f"{input_size} -> {' -> '.join(map(str, hidden_layers))} -> {output_size}"

    def forward(self, x):
        """Forward pass through the network"""
        # x: input state tensor
        # Apply ReLU activation to each hidden layer
        for layer in self.layers:
            x = F.relu(layer(x))
        # Final layer has no activation - outputs raw Q-values
        x = self.output_layer(x)
        return x

    def save(self, file_name='model.pth'):
        """Save the model to file"""
        # Directory to store saved models
        model_folder_path = './model'
        # Create directory if it doesn't exist
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # Build full file path
        file_name = os.path.join(model_folder_path, file_name)
        # Save model weights and biases
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        """Load model from file if it exists"""
        # Directory where models are stored
        model_folder_path = './model'
        # Build full file path
        file_name = os.path.join(model_folder_path, file_name)
        # Check if model file exists before loading
        if os.path.exists(file_name):
            # Load saved weights and biases into model
            self.load_state_dict(torch.load(file_name))
            # Set model to evaluation mode (disables dropout, etc.)
            self.eval()
            print(f"Loaded trained model from {file_name}")
            return True
        return False


# ===============================================
# 3. TRAINER CLASS
# ===============================================
# Q-learning trainer that implements the Bellman equation to optimize the neural network using experience replay.

class QTrainer:
    """Trainer class for the Q-learning neural network"""

    def __init__(self, model, lr, gamma):
        # lr: learning rate for optimizer
        # gamma: discount factor for future rewards (0-1)
        # model: the Q-network to train
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # Adam optimizer for gradient descent
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Mean Squared Error loss for Q-learning
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """Perform one training step with gradient descent"""
        # state: current state(s)
        # action: action(s) taken
        # reward: reward(s) received
        # next_state: resulting state(s)
        # done: whether episode(s) ended

        # Convert inputs to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Handle single values vs batches - add batch dimension if needed
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Get predicted Q-values for current states
        pred = self.model(state)

        # Clone predictions to create target Q-values
        target = pred.clone()
        # Update Q-values using Bellman equation
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Bellman equation: Q(s,a) = r + gamma * max(Q(s',a'))
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Update target Q-value for the action taken
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Perform gradient descent
        self.optimizer.zero_grad()  # Clear previous gradients
        loss = self.criterion(target, pred)  # Calculate MSE loss
        loss.backward()  # Backpropagation
        self.optimizer.step()  # Update weights
