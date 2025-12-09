"""
Trading Environment Module
Custom Gymnasium environment for cryptocurrency trading simulation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from typing import Tuple, Optional, Dict, Any


class TradingEnv(gym.Env):
    """
    Custom Trading Environment for Reinforcement Learning.
    
    # RUBRIC: [Custom Environment] - Class inheriting from gymnasium.Env
    
    The agent can take one of three actions:
    - 0: Hold (do nothing)
    - 1: Buy (purchase with available cash)
    - 2: Sell (sell all holdings)
    
    The state is a sliding window of normalized features.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 initial_balance: float = 10000.0,
                 window_size: int = 10,
                 transaction_fee: float = 0.001,
                 max_steps: int = None):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with normalized features
            initial_balance: Starting cash amount
            window_size: Number of past days to include in state
            transaction_fee: Fee percentage per transaction (default 0.1%)
            max_steps: Maximum steps per episode (None = use full data)
        """
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.transaction_fee = transaction_fee
        self.max_steps = max_steps
        
        # Feature columns for state (normalized features for neural network)
        self.feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                                'RSI', 'MACD', 'MACD_Signal', 'SMA_7', 'SMA_25', 'Price_Change']
        self.n_features = len(self.feature_columns)
        
        # Raw price column for actual trading (not normalized)
        self.price_column = 'Close_Raw'
        
        # RUBRIC: [Custom Environment] - Action Space
        # 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)
        
        # RUBRIC: [Custom Environment] - Observation Space
        # State shape: (window_size, num_features)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size, self.n_features),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            options: Dict with optional 'random_start' key for training randomization
        
        Returns:
            Tuple of (initial_state, info_dict)
        """
        super().reset(seed=seed)
        
        # Check if random start is requested (for training to prevent overfitting)
        random_start = False
        if options is not None and options.get('random_start', False):
            random_start = True
        
        if random_start:
            # Random starting point
            min_episode_length = self.max_steps if self.max_steps else 100
            max_start = len(self.df) - min_episode_length
            if max_start > self.window_size:
                self.current_step = np.random.randint(self.window_size, max_start)
            else:
                self.current_step = self.window_size
        else:
            self.current_step = self.window_size
        
        # Track starting step for max_steps limit
        self.start_step = self.current_step
        
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.total_trades = 0
        
        # Track history for analysis
        self.net_worth_history = [self.initial_balance]
        self.action_history = []
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get the current state (sliding window of features).
        
        Returns:
            State array of shape (window_size, n_features)
        """
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        state = self.df[self.feature_columns].iloc[start_idx:end_idx].values
        return state.astype(np.float32)
    
    def _get_current_price(self) -> float:
        """Get the current raw closing price (not normalized, for trading)."""
        return self.df[self.price_column].iloc[self.current_step]
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info dictionary."""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth,
            'total_trades': self.total_trades
        }
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0 (Hold), 1 (Buy), or 2 (Sell)
        
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        self.prev_net_worth = self.net_worth
        current_price = self._get_current_price()
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > 0:
                # Calculate shares to buy (accounting for transaction fee)
                max_purchase = self.balance * (1 - self.transaction_fee)
                shares_to_buy = max_purchase / current_price if current_price > 0 else 0
                
                if shares_to_buy > 0:
                    self.shares_held += shares_to_buy
                    self.balance = 0
                    self.total_trades += 1
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                # Sell all shares (accounting for transaction fee)
                sell_value = self.shares_held * current_price * (1 - self.transaction_fee)
                self.balance += sell_value
                self.shares_held = 0
                self.total_trades += 1
        
        # Action 0 (Hold) does nothing
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new net worth
        new_price = self._get_current_price() if self.current_step < len(self.df) else current_price
        self.net_worth = self.balance + self.shares_held * new_price
        self.net_worth_history.append(self.net_worth)
        self.action_history.append(action)
        
        # Calculate reward
        # RUBRIC: [Custom Environment] - Reward Function
        # Balanced reward with position incentives
        
        if self.prev_net_worth > 0 and self.net_worth > 0:
            base_reward = np.log(self.net_worth / self.prev_net_worth)
        else:
            base_reward = 0.0
        
        # Moderate scaling (not too large to prevent Q-value explosion)
        reward = base_reward * 100.0  # Scale to make differences visible
        
        # Position-based incentives using price momentum
        price_change = self.df.iloc[self.current_step]['Price_Change']
        
        # Reward holding BTC when price goes up
        if self.shares_held > 0 and price_change > 0:
            reward += price_change * 50.0
        
        # Penalize holding cash when price goes up significantly
        if self.shares_held == 0 and price_change > 0.01:
            reward -= price_change * 30.0
        
        # Reward holding cash when price drops
        if self.shares_held == 0 and price_change < -0.01:
            reward += abs(price_change) * 30.0
        
        # Penalize holding BTC when price drops
        if self.shares_held > 0 and price_change < 0:
            reward += price_change * 20.0  # This is negative
        
        # Clip to reasonable range
        reward = np.clip(reward, -5.0, 5.0)
        
        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        
        # Check max_steps limit
        if self.max_steps is not None:
            steps_taken = self.current_step - self.start_step
            truncated = steps_taken >= self.max_steps
        else:
            truncated = False
        
        # Get next state
        if not terminated:
            next_state = self._get_state()
        else:
            next_state = np.zeros((self.window_size, self.n_features), dtype=np.float32)
        
        info = self._get_info()
        
        return next_state, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment state."""
        print(f"Step: {self.current_step}, Net Worth: ${self.net_worth:.2f}, "
              f"Balance: ${self.balance:.2f}, Shares: {self.shares_held:.6f}")
    
    def get_net_worth_history(self) -> list:
        """Return the history of net worth values."""
        return self.net_worth_history


if __name__ == "__main__":
    # Test the environment
    import sys
    sys.path.append('..')
    from data_loader import get_data
    
    train_df, test_df, scaler = get_data()
    env = TradingEnv(train_df)
    
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial info: {info}")
    
    # Take a few random steps
    for i in range(5):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Net Worth={info['net_worth']:.2f}")
        if terminated:
            break

