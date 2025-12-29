import numpy as np


def mse(y_pred: np.ndarray, y: np.ndarray) -> float:
    e = y_pred - y
    return float(np.mean(e * e))


def mae(y_pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y)))


def accuracy(y_pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y_pred == y).astype(np.float64)))


def confusion_matrix_binary(y_pred: np.ndarray, y: np.ndarray):
    tp = int(np.sum((y_pred == 1) & (y == 1)))
    tn = int(np.sum((y_pred == 0) & (y == 0)))
    fp = int(np.sum((y_pred == 1) & (y == 0)))
    fn = int(np.sum((y_pred == 0) & (y == 1)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
