"""
Main Execution Script for RL Crypto Trader
Trains a DQN agent to trade Bitcoin and evaluates performance.
"""

import os
import sys
import numpy as np
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import get_data, save_data
from environment import TradingEnv
from agent import DQNAgent
from utils import (
    plot_training_curve, 
    plot_wealth_curve, 
    calculate_buy_hold_wealth,
    print_evaluation_summary
)

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Data settings
    'symbol': 'BTC-USD',
    'data_start': '2015-01-01',  # Training starts from 2015
    'data_end': '2023-12-31',    # Data ends at 2023
    'train_end': '2021-12-31',   # Test set: 2022-2023 (bear + bull recovery)
    
    # Environment settings
    'initial_balance': 10000.0,
    'window_size': 10,
    'transaction_fee': 0.001,
    
    # Agent settings
    'learning_rate': 0.0003,     # Lower learning rate for stability
    'gamma': 0.95,               # Lower gamma to reduce Q-value accumulation
    'epsilon': 1.0,
    'epsilon_min': 0.02,
    'epsilon_decay': 0.997,
    'buffer_size': 50000,
    'batch_size': 64,
    'target_update_freq': 10,    # Less frequent updates for stability
    
    # Training settings
    'episodes': 1500,            # Balanced training
    
    # Paths
    'data_dir': 'data',
    'results_dir': 'results',
    'models_dir': 'models',
}


def train_agent(env: TradingEnv, agent: DQNAgent, episodes: int) -> list:
    """
    Train the DQN agent.
    
    # RUBRIC: [Training Curves] - Log rewards over episodes
    
    Args:
        env: Trading environment
        agent: DQN agent
        episodes: Number of training episodes
    
    Returns:
        List of total rewards per episode
    """
    print("\n" + "="*60)
    print("PHASE 2: TRAINING")
    print("="*60)
    
    episode_rewards = []
    
    for episode in tqdm(range(1, episodes + 1), desc="Training"):
        # Use random start position to prevent overfitting
        state, info = env.reset(options={'random_start': True})
        total_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.act(state, training=True)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train on batch
            agent.replay()
            
            state = next_state
            total_reward += reward
        
        # End of episode updates
        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        
        # Update target network periodically
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        # Progress logging
        if episode % 10 == 0 or episode == 1:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"\nEpisode {episode}/{episodes} | "
                  f"Reward: {total_reward:.2f} | "
                  f"Avg(10): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Net Worth: ${info['net_worth']:.2f}")
    
    print("\nTraining completed!")
    return episode_rewards


def evaluate_agent(env: TradingEnv, agent: DQNAgent) -> tuple:
    """
    Evaluate the trained agent on test data.
    
    # RUBRIC: [Simulation-based Evaluation] - Run backtest on Test Set
    
    Args:
        env: Trading environment (initialized with test data)
        agent: Trained DQN agent
    
    Returns:
        Tuple of (agent_wealth_history, total_trades)
    """
    print("\n" + "="*60)
    print("PHASE 3: EVALUATION (BACKTEST)")
    print("="*60)
    
    # Set agent to inference mode (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    state, info = env.reset()
    done = False
    
    while not done:
        action = agent.act(state, training=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
    
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    wealth_history = env.get_net_worth_history()
    total_trades = info['total_trades']
    
    print(f"Backtest completed: {len(wealth_history)} steps, {total_trades} trades")
    
    return wealth_history, total_trades


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("RL CRYPTO TRADER - DQN Bitcoin Trading Agent")
    print("CS 372 Final Project")
    print("="*60)
    
    # Create directories
    for dir_name in [CONFIG['data_dir'], CONFIG['results_dir'], CONFIG['models_dir']]:
        os.makedirs(dir_name, exist_ok=True)
    
    # ============================================================
    # PHASE 1: SETUP - Load Data
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 1: DATA LOADING & PREPROCESSING")
    print("="*60)
    
    # RUBRIC: [Original Dataset] - Fetch raw data via API
    # RUBRIC: [Feature Engineering] - Calculate technical indicators
    # RUBRIC: [Preprocessing] - Normalize/Scale data
    train_df, test_df, scaler = get_data(
        symbol=CONFIG['symbol'],
        start=CONFIG['data_start'],
        end=CONFIG['data_end'],
        train_end=CONFIG['train_end']
    )
    
    # Save data for reference
    save_data(train_df, test_df, CONFIG['data_dir'])
    
    # Initialize environments
    # RUBRIC: [Custom Environment]
    train_env = TradingEnv(
        df=train_df,
        initial_balance=CONFIG['initial_balance'],
        window_size=CONFIG['window_size'],
        transaction_fee=CONFIG['transaction_fee'],
        max_steps=200  # Limit episode length for training stability
    )
    
    test_env = TradingEnv(
        df=test_df,
        initial_balance=CONFIG['initial_balance'],
        window_size=CONFIG['window_size'],
        transaction_fee=CONFIG['transaction_fee'],
        max_steps=None  # No limit for evaluation - run full test period
    )
    
    # Initialize agent
    # RUBRIC: [Custom Architecture]
    # RUBRIC: [Regularization]
    state_shape = (CONFIG['window_size'], len(train_env.feature_columns))
    agent = DQNAgent(
        state_shape=state_shape,
        action_size=3,
        learning_rate=CONFIG['learning_rate'],
        gamma=CONFIG['gamma'],
        epsilon=CONFIG['epsilon'],
        epsilon_min=CONFIG['epsilon_min'],
        epsilon_decay=CONFIG['epsilon_decay'],
        buffer_size=CONFIG['buffer_size'],
        batch_size=CONFIG['batch_size'],
        target_update_freq=CONFIG['target_update_freq']
    )
    
    print(f"\nEnvironment State Shape: {state_shape}")
    print(f"Action Space: {train_env.action_space}")
    
    # ============================================================
    # PHASE 2: TRAINING
    # ============================================================
    episode_rewards = train_agent(train_env, agent, CONFIG['episodes'])
    
    # Save model
    model_path = os.path.join(CONFIG['models_dir'], 'dqn_model.pth')
    agent.save(model_path)
    
    # ============================================================
    # PHASE 3: EVALUATION (BACKTEST)
    # ============================================================
    agent_wealth, total_trades = evaluate_agent(test_env, agent)
    
    # RUBRIC: [Baseline Comparison] - Calculate Buy & Hold baseline
    # Get the original (raw) Close prices from test data for baseline calculation
    test_prices = test_df['Close_Raw'].values[CONFIG['window_size']:]
    
    # For buy & hold, calculate wealth using raw prices
    baseline_wealth = calculate_buy_hold_wealth(
        prices=test_prices.tolist(),
        initial_balance=CONFIG['initial_balance']
    )
    
    # Ensure both arrays have the same length
    min_length = min(len(agent_wealth), len(baseline_wealth))
    agent_wealth = agent_wealth[:min_length]
    baseline_wealth = baseline_wealth[:min_length]
    
    # ============================================================
    # PHASE 4: VISUALIZATION
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 4: VISUALIZATION")
    print("="*60)
    
    # RUBRIC: [Training Curves] - Plot rewards/loss
    training_curve_path = os.path.join(CONFIG['results_dir'], 'training_curve.png')
    plot_training_curve(
        rewards=episode_rewards,
        losses=agent.get_losses(),
        save_path=training_curve_path
    )
    
    # RUBRIC: [Simulation-based Evaluation] - Plot Wealth Curve
    # RUBRIC: [Baseline Comparison] - Agent vs Buy & Hold
    wealth_curve_path = os.path.join(CONFIG['results_dir'], 'wealth_curve.png')
    plot_wealth_curve(
        agent_wealth=agent_wealth,
        baseline_wealth=baseline_wealth,
        save_path=wealth_curve_path
    )
    
    # Print summary
    print_evaluation_summary(agent_wealth, baseline_wealth, total_trades)
    
    print("\n" + "="*60)
    print("EXECUTION COMPLETED")
    print("="*60)
    print(f"\nOutputs saved to:")
    print(f"  - Model: {model_path}")
    print(f"  - Training Curve: {training_curve_path}")
    print(f"  - Wealth Curve: {wealth_curve_path}")
    print(f"  - Data: {CONFIG['data_dir']}/")


if __name__ == "__main__":
    main()

