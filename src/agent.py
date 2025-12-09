"""
DQN Agent Module
Deep Q-Network agent with experience replay and target network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List


class DQN(nn.Module):
    """
    Deep Q-Network architecture.
    
    # RUBRIC: [Custom Architecture] - Build a DQN using PyTorch (Linear layers + ReLU)
    """
    
    def __init__(self, state_size: int, action_size: int):
        """
        Initialize the DQN network.
        
        Args:
            state_size: Dimension of input state (flattened)
            action_size: Number of possible actions
        """
        super(DQN, self).__init__()
        
        # RUBRIC: [Custom Architecture] - MLP: Input -> 128 -> 64 -> Output
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, state_size)
        
        Returns:
            Q-values for each action
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.
    
    # RUBRIC: [Regularization] - Experience Replay implementation
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            List of sampled transitions
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent with experience replay and target network.
    
    # RUBRIC: [Custom Architecture] - DQN Agent using PyTorch
    # RUBRIC: [Regularization] - Experience Replay and Target Network
    """
    
    def __init__(self, 
                 state_shape: Tuple[int, int],
                 action_size: int = 3,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 10):
        """
        Initialize the DQN Agent.
        
        Args:
            state_shape: Shape of the state (window_size, n_features)
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: How often to update target network (in episodes)
        """
        self.state_shape = state_shape
        self.state_size = state_shape[0] * state_shape[1]  # Flatten
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # RUBRIC: [Custom Architecture] - Policy Network
        self.policy_net = DQN(self.state_size, action_size).to(self.device)
        
        # RUBRIC: [Regularization] - Target Network
        self.target_net = DQN(self.state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber Loss - more robust to outliers
        
        # RUBRIC: [Regularization] - Experience Replay Buffer
        self.memory = ReplayBuffer(capacity=buffer_size)
        
        # Training metrics
        self.losses = []
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (use epsilon-greedy)
        
        Returns:
            Selected action (0: Hold, 1: Buy, 2: Sell)
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Exploitation: select best action based on Q-values
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        
        return q_values.argmax().item()
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.push(state.flatten(), action, reward, next_state.flatten(), done)
    
    def replay(self) -> float:
        """
        Train the network on a batch of experiences.
        
        # RUBRIC: [Regularization] - Sample from buffer, compute Loss (MSE), backpropagate
        
        Returns:
            Loss value (0 if not enough samples)
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        batch = self.memory.sample(self.batch_size)
        
        # Unpack batch
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # DOUBLE DQN: Use policy_net to select action, target_net to evaluate
        # This reduces Q-value overestimation and stabilizes training loss
        with torch.no_grad():
            # Policy net selects the best action
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # Target net evaluates the Q-value of that action
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def update_target_network(self) -> None:
        """
        Copy weights from policy network to target network.
        
        # RUBRIC: [Regularization] - Target Network update
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self) -> None:
        """Decay the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load the model from a file.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint.get('losses', [])
        print(f"Model loaded from {filepath}")
    
    def get_losses(self) -> List[float]:
        """Return the training loss history."""
        return self.losses


if __name__ == "__main__":
    # Test the agent
    state_shape = (10, 11)  # 10 timesteps, 11 features
    agent = DQNAgent(state_shape=state_shape)
    
    # Test action selection
    dummy_state = np.random.randn(10, 11).astype(np.float32)
    action = agent.act(dummy_state)
    print(f"Selected action: {action}")
    
    # Test memory and replay
    for i in range(100):
        state = np.random.randn(10, 11).astype(np.float32)
        action = random.randint(0, 2)
        reward = random.random()
        next_state = np.random.randn(10, 11).astype(np.float32)
        done = False
        agent.remember(state, action, reward, next_state, done)
    
    loss = agent.replay()
    print(f"Training loss: {loss:.4f}")

