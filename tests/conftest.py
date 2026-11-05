"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs.

    Yields:
        Path to a temporary directory that is cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_results():
    """Provide sample experiment results for testing.

    Returns:
        Dictionary with typical experiment result structure.
    """
    return {
        "config": {
            "n_samples": 100,
            "learning_rate": 0.01,
            "epochs": 50,
        },
        "metrics": {
            "mse": 0.05,
            "rmse": 0.224,
            "r2": 0.95,
        },
        "model_params": {
            "slope": 2.5,
            "intercept": -1.0,
        },
    }
