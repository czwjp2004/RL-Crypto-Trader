# Attribution

## Data Sources

### Primary Data
* **Yahoo Finance API (`yfinance`)**: Used for fetching historical OHLCV (Open, High, Low, Close, Volume) data for Bitcoin (BTC-USD).
  - Documentation: https://pypi.org/project/yfinance/
  - License: Apache License 2.0
  - Data Range: 2015-01-01 to 2024-12-31
  - No API key required, freely available for educational use

## Libraries and Frameworks

### Core Dependencies

* **Gymnasium (OpenAI Gym)**: Used as the standard framework for the Reinforcement Learning environment interface.
  - Version: Latest (1.0.0+)
  - License: MIT
  - Purpose: Provides `gym.Env` base class and standardized API for RL environments
  - Documentation: https://gymnasium.farama.org/

* **PyTorch**: Used for implementing the Deep Q-Network (DQN) architecture.
  - Version: Latest (2.0.0+)
  - License: BSD-style
  - Purpose: Neural network implementation, automatic differentiation, GPU acceleration
  - Components Used: `torch.nn`, `torch.optim`, `torch.Tensor`
  - Documentation: https://pytorch.org/

* **NumPy**: Used for numerical computations and array operations.
  - License: BSD
  - Purpose: State representation, reward calculations, mathematical operations

* **Pandas**: Used for data manipulation and CSV I/O.
  - License: BSD
  - Purpose: DataFrames for time-series data, feature engineering

* **Matplotlib**: Used for generating training curves and portfolio performance plots.
  - License: PSF-based
  - Purpose: Visualization of results (training reward/loss, wealth curves)

* **scikit-learn**: Used for data preprocessing (normalization).
  - License: BSD
  - Purpose: `StandardScaler` for Z-score normalization of features

* **tqdm**: Used for progress bars during training.
  - License: MIT/MPL
  - Purpose: User-friendly training progress visualization

## Algorithm References

### Deep Q-Network (DQN)
* **Original Paper**: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
  - arXiv: https://arxiv.org/abs/1312.5602
  - Techniques Implemented: Experience replay, target networks

* **Double DQN**: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
  - arXiv: https://arxiv.org/abs/1509.06461
  - Purpose: Reduces Q-value overestimation bias
  - Implementation: Target network used for action evaluation

### Technical Analysis Indicators
* **RSI (Relative Strength Index)**: Standard 14-period RSI formula
  - Reference: J. Welles Wilder Jr., "New Concepts in Technical Trading Systems" (1978)
  
* **MACD (Moving Average Convergence Divergence)**: Standard 12/26/9 EMA configuration
  - Reference: Gerald Appel, "Technical Analysis: Power Tools for Active Investors" (2005)

## AI Assistance Declaration

As per CS 372 course policy, the following AI tools were used in the development of this project:

### AI-Assisted Code Generation

### Specific AI-Generated Components

1. **Environment Boilerplate** (`src/environment.py`):
   - Gymnasium interface implementation (Cursor-generated)
   - Modified reward function logic (human-designed with AI refinement)

2. **Plotting Utilities** (`src/utils.py`):
   - `plot_training_curve()` and `plot_wealth_curve()` functions (Cursor-generated)
   - Minor adjustments made manually for styling

3. **Documentation**:
   - README template structure (AI-assisted)
   - Docstrings and comments (mixed human/AI)
   - This ATTRIBUTION.md file structure (AI template)

### AI Usage for Problem-Solving

* **Brainstorming**: Used Claude to explore different reward function designs (log returns, Sharpe ratio, asymmetric penalties)
* **Debugging**: AI helped identify that loss divergence was caused by excessive reward scaling
* **Optimization**: Suggestions for gradient clipping and learning rate schedules
* **Testing**: AI-generated test cases for environment state transitions

### Human Contributions

All core design decisions and implementation were made by the human developer:

**1. Core Algorithm Design:**
- Project architecture and module organization
- Selection of Deep Q-Network with Double DQN enhancement
- Complete implementation of trading environment and agent logic
- Integration of yfinance API with Gymnasium framework

**2. Feature Engineering:**
- Technical indicators: RSI, MACD, Simple Moving Averages
- State representation: 10-timestep rolling window (110 dimensions)
- Z-score normalization and data preprocessing pipeline
- Train/test split strategy: 2015-2021 train, 2022-2023 test

**3. Reward Function Design (Critical Innovation):**
- Developed position-aware reward through 5+ iterations
- Final design: Log return base + momentum-based incentives
  - Rewards holding BTC during uptrends (+50x price change)
  - Penalizes cash holdings during rallies (-30x price change)
  - Rewards cash preservation during crashes (+30x |drop|)
- Extensive experimentation to balance exploration vs. exploitation

**4. Hyperparameter Optimization:**
- Learning rate: 0.0003, Gamma: 0.95, Epsilon decay: 0.997
- Batch size: 64, Buffer: 50K, Target update: every 10 episodes
- Network architecture: 128→64 hidden layers
- All parameters tuned through systematic trial (15+ training runs)

**5. Problem-Solving & Debugging:**
- Diagnosed and fixed Q-value explosion (reduced reward scaling)
- Resolved loss divergence (adjusted gamma from 0.99 to 0.95)
- Implemented gradient clipping and Huber Loss for stability
- Iteratively improved agent from -15% to +27% test performance

**6. Evaluation Framework:**
- Backtesting implementation with realistic transaction fees
- Metrics: Total return, max drawdown, trade frequency
- Visualization: Training curves and portfolio comparison plots
- Comprehensive documentation and reproducibility setup

## Intellectual Property

* **Original Work**: The overall project architecture, reward function design, and integration of components is original work by the author.
* **Derived Work**: The DQN algorithm implementation closely follows standard PyTorch tutorials and research papers (cited above).
* **No Plagiarism**: All code written for this project is either:
  1. Original implementation by the author
  2. AI-generated with explicit acknowledgment (this document)
  3. Derived from cited open-source libraries

## Course Compliance

This project complies with CS 372 course policies:
- AI usage is fully disclosed
- All external libraries are cited
- Data sources are documented
- Algorithm references are provided
- Collaboration (with AI) is acknowledged

## Contact

**Author:** Junpeng Wang(czwjp2004) and Yichen Jia
**GitHub:** https://github.com/czwjp2004/RL-Crypto-Trader

For questions about attributions or to report missing citations:
- Course: CS 372
- Institution: Duke University
- Semester: Fall 2025
- Instructor: Brandon Fain

---

**Last Updated**: December 2025

