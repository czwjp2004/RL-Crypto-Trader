# Setup Instructions

## Prerequisites
* Python 3.8+ (Tested on Python 3.11)
* Internet connection (for `yfinance` API to fetch Bitcoin data)
* ~500MB disk space (for downloaded data and saved models)

## Installation Steps

### 1. Environment Setup
It is **strongly recommended** to use a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

**Package List:**
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `yfinance` - Yahoo Finance API for Bitcoin data
- `gymnasium` - Reinforcement Learning environment framework
- `torch` - PyTorch for Deep Q-Network
- `scikit-learn` - Preprocessing utilities
- `tqdm` - Progress bars

**Note:** Installation may take 2-5 minutes depending on your internet speed and whether PyTorch needs to download CUDA libraries.

### 3. Verify Installation

Test that all components are working correctly:

#### Step 3.1: Test Data Fetching
```bash
python -c "import yfinance as yf; print(yf.download('BTC-USD', start='2024-01-01', end='2024-01-02'))"
```
You should see Bitcoin price data printed.

#### Step 3.2: Test PyTorch
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```
You should see the PyTorch version printed (e.g., `2.0.0` or higher).

#### Step 3.3: Test Gymnasium
```bash
python -c "import gymnasium as gym; print('Gymnasium working!')"
```
You should see "Gymnasium working!" printed.

### 4. Run the Complete Pipeline

Execute the main training and evaluation script:

```bash
python main.py
```

**Expected Output:**
1. **Phase 1:** Data fetching and preprocessing (10-20 seconds)
   - Downloads Bitcoin data from 2015-2023
   - Calculates technical indicators (RSI, MACD, SMA)
   - Splits into training (2015-2021) and test (2022-2023) sets

2. **Phase 2:** DQN Training (3-5 minutes)
   - Progress bar showing 1500 episodes
   - Periodic episode summaries (every 100 episodes)
   - Saved model to `models/dqn_model.pth`

3. **Phase 3:** Backtesting (5-10 seconds)
   - Evaluates trained agent on test data
   - Compares against Buy & Hold baseline

4. **Phase 4:** Visualization (2-3 seconds)
   - Generates `results/training_curve.png`
   - Generates `results/wealth_curve.png`

**Output Files:**
```
data/
  ├── train_data.csv
  └── test_data.csv
models/
  └── dqn_model.pth
results/
  ├── training_curve.png
  └── wealth_curve.png
```

### 5. Troubleshooting

#### Issue 1: `ModuleNotFoundError: No module named 'yfinance'`
**Solution:** Ensure you've activated the virtual environment and run `pip install -r requirements.txt`

#### Issue 2: `RuntimeError: CUDA out of memory`
**Solution:** The code defaults to CPU mode. If you explicitly enabled CUDA, reduce batch size in `main.py`:
```python
CONFIG = {
    'batch_size': 32,  # Reduce from 64
}
```

#### Issue 3: `yfinance` download hangs or times out
**Solution:** Check your internet connection. If behind a firewall/proxy, set environment variables:
```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

#### Issue 4: Training is very slow
**Solution:** 
- Reduce episodes in `main.py`: `'episodes': 500`
- Ensure no other heavy processes are running
- On Mac M1/M2, PyTorch should automatically use Metal acceleration

#### Issue 5: `FileNotFoundError` for results directory
**Solution:** The script auto-creates directories, but if needed:
```bash
mkdir -p data models results
```

## 6. Running Individual Components

### Data Loading Only
```bash
python -c "from src.data_loader import get_data, save_data; df = get_data('BTC-USD', '2020-01-01', '2023-01-01'); save_data(df, 'data/')"
```

### Training Only (Resume from Checkpoint)
```bash
python -c "from main import train_agent; # ... (see main.py for implementation)"
```

### Evaluation Only (Using Pre-trained Model)
```bash
# Modify main.py to skip training and load existing model
# Set CONFIG['episodes'] = 0 and uncomment model loading
```

## 7. Development Setup (Optional)

For code development and debugging:

### Install Additional Dev Dependencies
```bash
pip install jupyter ipython pytest
```

### Run Tests (if implemented)
```bash
pytest tests/
```

### Launch Jupyter Notebook for Exploration
```bash
jupyter notebook
```

## 8. Hardware Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4 GB
- Storage: 500 MB

**Recommended:**
- CPU: 4+ cores
- RAM: 8 GB
- Storage: 2 GB

**Training Time Estimates:**
- Minimum setup: ~8 minutes (1500 episodes)
- Recommended setup: ~3 minutes (1500 episodes)
- With GPU (CUDA): ~2 minutes (1500 episodes)

## 9. Platform-Specific Notes

### macOS
- Works natively on Intel and Apple Silicon (M1/M2)
- PyTorch automatically uses Metal backend on M1/M2
- No additional configuration needed

### Windows
- Tested on Windows 10/11
- Use PowerShell or Command Prompt
- Some `yfinance` network issues may occur; try running as Administrator

### Linux
- Tested on Ubuntu 20.04+
- Ensure Python 3.8+ is installed: `python3 --version`
- May need to install system dependencies: `sudo apt-get install python3-dev`

## 10. Next Steps

After successful installation:
1. Review `README.md` for project overview
2. Run `python main.py` to reproduce results
3. Experiment with hyperparameters in `main.py`
4. View results in `results/` directory

For questions or issues, please open an issue on the repository or contact the maintainer.

