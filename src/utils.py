"""
Utility Functions Module
Helper functions for plotting and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import os


def plot_training_curve(rewards: List[float], 
                        losses: Optional[List[float]] = None,
                        save_path: str = "results/training_curve.png",
                        show: bool = False) -> None:
    """
    Plot training rewards and losses over episodes.
    
    # RUBRIC: [Training Curves] - Log and plot rewards/loss over episodes
    
    Args:
        rewards: List of total rewards per episode
        losses: Optional list of average losses per episode
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if losses is not None and len(losses) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    
    # Plot rewards
    episodes = range(1, len(rewards) + 1)
    ax1.plot(episodes, rewards, 'b-', linewidth=1.5, label='Episode Reward')
    
    # Add moving average
    window = min(10, len(rewards))
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(rewards) + 1), moving_avg, 'r-', 
                linewidth=2, label=f'{window}-Episode Moving Avg')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Training Reward Over Episodes', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot losses if provided
    if losses is not None and len(losses) > 0:
        ax2.plot(range(1, len(losses) + 1), losses, 'g-', linewidth=1, alpha=0.7)
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Loss (MSE)', fontsize=12)
        ax2.set_title('Training Loss Over Time', fontsize=14)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curve saved to {save_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_wealth_curve(agent_wealth: List[float],
                      baseline_wealth: List[float],
                      save_path: str = "results/wealth_curve.png",
                      show: bool = False) -> None:
    """
    Plot wealth curves comparing agent performance vs baseline.
    
    # RUBRIC: [Simulation-based Evaluation] - Plot the Wealth Curve
    # RUBRIC: [Baseline Comparison] - Compare Agent vs. Buy & Hold
    
    Args:
        agent_wealth: List of agent's net worth over time
        baseline_wealth: List of buy & hold net worth over time
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    steps = range(len(agent_wealth))
    
    # Plot agent wealth
    ax.plot(steps, agent_wealth, 'b-', linewidth=2, label='DQN Agent')
    
    # Plot baseline wealth
    ax.plot(steps, baseline_wealth, 'r--', linewidth=2, label='Buy & Hold Baseline')
    
    # Calculate and display final returns
    agent_return = (agent_wealth[-1] - agent_wealth[0]) / agent_wealth[0] * 100
    baseline_return = (baseline_wealth[-1] - baseline_wealth[0]) / baseline_wealth[0] * 100
    
    ax.set_xlabel('Trading Step', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_title('Portfolio Performance: DQN Agent vs Buy & Hold Strategy', fontsize=14)
    
    # Add annotations
    textstr = f'DQN Agent Return: {agent_return:+.2f}%\nBuy & Hold Return: {baseline_return:+.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at initial investment
    ax.axhline(y=agent_wealth[0], color='gray', linestyle=':', alpha=0.5, label='Initial Investment')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Wealth curve saved to {save_path}")
    
    if show:
        plt.show()
    plt.close()


def calculate_buy_hold_wealth(prices: List[float], 
                               initial_balance: float = 10000.0) -> List[float]:
    """
    Calculate the wealth history for a buy and hold strategy.
    
    # RUBRIC: [Baseline Comparison] - Buy & Hold strategy calculation
    
    Args:
        prices: List of asset prices
        initial_balance: Initial investment amount
    
    Returns:
        List of portfolio values over time
    """
    if len(prices) == 0:
        return [initial_balance]
    
    # Buy at the first price
    initial_price = prices[0]
    shares = initial_balance / initial_price
    
    # Calculate wealth at each step
    wealth_history = [shares * price for price in prices]
    
    return wealth_history


def print_evaluation_summary(agent_wealth: List[float],
                              baseline_wealth: List[float],
                              total_trades: int) -> None:
    """
    Print a summary of the evaluation results.
    
    Args:
        agent_wealth: Agent's wealth history
        baseline_wealth: Baseline wealth history
        total_trades: Total number of trades made
    """
    initial = agent_wealth[0]
    
    agent_final = agent_wealth[-1]
    baseline_final = baseline_wealth[-1]
    
    agent_return = (agent_final - initial) / initial * 100
    baseline_return = (baseline_final - initial) / initial * 100
    
    agent_max = max(agent_wealth)
    agent_min = min(agent_wealth)
    baseline_max = max(baseline_wealth)
    baseline_min = min(baseline_wealth)
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nInitial Investment: ${initial:,.2f}")
    print(f"\nDQN Agent Performance:")
    print(f"  - Final Value:    ${agent_final:,.2f}")
    print(f"  - Total Return:   {agent_return:+.2f}%")
    print(f"  - Max Value:      ${agent_max:,.2f}")
    print(f"  - Min Value:      ${agent_min:,.2f}")
    print(f"  - Total Trades:   {total_trades}")
    print(f"\nBuy & Hold Baseline:")
    print(f"  - Final Value:    ${baseline_final:,.2f}")
    print(f"  - Total Return:   {baseline_return:+.2f}%")
    print(f"  - Max Value:      ${baseline_max:,.2f}")
    print(f"  - Min Value:      ${baseline_min:,.2f}")
    print(f"\nRelative Performance:")
    if baseline_return != 0:
        outperformance = agent_return - baseline_return
        print(f"  - Agent vs Baseline: {outperformance:+.2f}% difference")
    if agent_final > baseline_final:
        print("  - Result: DQN Agent OUTPERFORMED Buy & Hold")
    else:
        print("  - Result: Buy & Hold outperformed DQN Agent")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test plotting functions
    import random
    
    # Generate dummy data
    rewards = [random.uniform(-100, 200) for _ in range(50)]
    losses = [random.uniform(0, 1) for _ in range(1000)]
    
    agent_wealth = [10000]
    baseline_wealth = [10000]
    
    for i in range(100):
        agent_wealth.append(agent_wealth[-1] * (1 + random.uniform(-0.02, 0.025)))
        baseline_wealth.append(baseline_wealth[-1] * (1 + random.uniform(-0.015, 0.02)))
    
    # Test plots
    plot_training_curve(rewards, losses, save_path="results/test_training.png")
    plot_wealth_curve(agent_wealth, baseline_wealth, save_path="results/test_wealth.png")
    print_evaluation_summary(agent_wealth, baseline_wealth, total_trades=25)

