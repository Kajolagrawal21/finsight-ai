"""
FinSight AI — Feature Engineering Tests
"""

import pytest
import pandas as pd
import numpy as np


def test_compute_indicators():
    """Test that technical indicators are computed correctly."""
    from data.processing.feature_engineering import compute_indicators

    # Create sample OHLCV data
    dates = pd.date_range("2024-01-01", periods=300, freq="1min")
    df = pd.DataFrame({
        "open":   np.random.uniform(100, 200, 300),
        "high":   np.random.uniform(150, 250, 300),
        "low":    np.random.uniform(80, 150, 300),
        "close":  np.random.uniform(100, 200, 300),
        "volume": np.random.randint(1000, 100000, 300),
    }, index=dates)

    result = compute_indicators(df)

    assert "rsi_14" in result.columns
    assert "macd" in result.columns
    assert "bb_upper" in result.columns
    assert "ema_20" in result.columns
    assert "atr_14" in result.columns
    assert len(result) > 0


def test_rsi_range():
    """Test RSI is always between 0 and 100."""
    from data.processing.feature_engineering import compute_indicators

    dates = pd.date_range("2024-01-01", periods=300, freq="1min")
    prices = np.random.uniform(100, 200, 300)
    df = pd.DataFrame({
        "open": prices, "high": prices * 1.01,
        "low": prices * 0.99, "close": prices,
        "volume": np.ones(300) * 1000,
    }, index=dates)

    result = compute_indicators(df)
    rsi = result["rsi_14"].dropna()

    assert (rsi >= 0).all(), "RSI should be >= 0"
    assert (rsi <= 100).all(), "RSI should be <= 100"


def test_bollinger_bands_logic():
    """Test Bollinger Bands: upper >= middle >= lower."""
    from data.processing.feature_engineering import compute_indicators

    dates = pd.date_range("2024-01-01", periods=300, freq="1min")
    prices = np.random.uniform(100, 200, 300)
    df = pd.DataFrame({
        "open": prices, "high": prices * 1.01,
        "low": prices * 0.99, "close": prices,
        "volume": np.ones(300) * 1000,
    }, index=dates)

    result = compute_indicators(df).dropna()

    assert (result["bb_upper"] >= result["bb_middle"]).all()
    assert (result["bb_middle"] >= result["bb_lower"]).all()
