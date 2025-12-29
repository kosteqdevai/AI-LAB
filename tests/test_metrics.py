import numpy as np

from ai_lab.metrics import accuracy, mae, mse


def test_mse_mae_basic():
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])  # errors: [1,0,-1]
    assert mse(y_pred, y) == (1.0 + 0.0 + 1.0) / 3.0
    assert mae(y_pred, y) == (1.0 + 0.0 + 1.0) / 3.0


def test_accuracy_basic():
    y = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert accuracy(y_pred, y) == 0.75
