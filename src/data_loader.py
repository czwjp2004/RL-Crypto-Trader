"""
Data Loader Module
Handles data fetching from yfinance API and feature engineering.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    # RUBRIC: [Feature Engineering] - RSI calculation
    
    Args:
        prices: Series of closing prices
        period: RSI period (default 14)
    
    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, 
                   fast: int = 12, 
                   slow: int = 26, 
                   signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate MACD and Signal Line.
    
    # RUBRIC: [Feature Engineering] - MACD calculation
    
    Args:
        prices: Series of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        Tuple of (MACD line, Signal line)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    # RUBRIC: [Feature Engineering] - Moving Average calculation
    
    Args:
        prices: Series of closing prices
        period: SMA period
    
    Returns:
        Series of SMA values
    """
    return prices.rolling(window=period).mean()


def get_data(symbol: str = "BTC-USD", 
             start: str = "2020-01-01", 
             end: str = "2024-12-31",
             train_end: str = "2022-12-31") -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fetch data from yfinance and perform feature engineering.
    
    # RUBRIC: [Original Dataset] - Fetch raw data via API and clean it programmatically
    # RUBRIC: [Preprocessing] - Normalize/Scale data using Z-score
    
    Args:
        symbol: Ticker symbol (default "BTC-USD")
        start: Start date for data
        end: End date for data
        train_end: Last date for training data (data after this is test)
    
    Returns:
        Tuple of (train_df, test_df, scaler)
    """
    print(f"Fetching data for {symbol} from {start} to {end}...")
    
    # Download data from yfinance
    # RUBRIC: [Original Dataset]
    df = yf.download(symbol, start=start, end=end, progress=False)
    
    if df.empty:
        raise ValueError(f"No data retrieved for {symbol}")
    
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Keep only necessary columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # RUBRIC: [Feature Engineering]
    # Calculate technical indicators
    df['RSI'] = calculate_rsi(df['Close'], period=14)
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'], fast=12, slow=26, signal=9)
    df['SMA_7'] = calculate_sma(df['Close'], period=7)
    df['SMA_25'] = calculate_sma(df['Close'], period=25)
    
    # Calculate price change percentage
    df['Price_Change'] = df['Close'].pct_change()
    
    # RUBRIC: [Preprocessing] - Drop NaNs
    df = df.dropna()
    
    print(f"Data shape after cleaning: {df.shape}")
    
    # Split into train and test
    train_df = df[df.index <= train_end].copy()
    test_df = df[df.index > train_end].copy()
    
    print(f"Training data: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} rows)")
    print(f"Test data: {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} rows)")
    
    # Save original Close price for trading (before normalization)
    train_df['Close_Raw'] = train_df['Close'].copy()
    test_df['Close_Raw'] = test_df['Close'].copy()
    
    # Define feature columns for scaling (exclude Close_Raw)
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'RSI', 'MACD', 'MACD_Signal', 'SMA_7', 'SMA_25', 'Price_Change']
    
    # RUBRIC: [Preprocessing] - Z-score normalization
    scaler = StandardScaler()
    
    # Fit scaler on training data only
    scaler.fit(train_df[feature_columns])
    
    # Transform both train and test
    train_scaled = scaler.transform(train_df[feature_columns])
    test_scaled = scaler.transform(test_df[feature_columns])
    
    # Create new dataframes with scaled features
    for i, col in enumerate(feature_columns):
        train_df[col] = train_scaled[:, i]
        test_df[col] = test_scaled[:, i]
    
    return train_df, test_df, scaler


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_dir: str = "data") -> None:
    """
    Save dataframes to CSV files.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        data_dir: Directory to save files
    """
    import os
    os.makedirs(data_dir, exist_ok=True)
    
    train_df.to_csv(f"{data_dir}/train_data.csv")
    test_df.to_csv(f"{data_dir}/test_data.csv")
    print(f"Data saved to {data_dir}/")


if __name__ == "__main__":
    # Test the data loader
    train_df, test_df, scaler = get_data()
    print("\nTrain DataFrame head:")
    print(train_df.head())
    print("\nTest DataFrame head:")
    print(test_df.head())

