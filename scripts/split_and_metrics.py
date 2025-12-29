import argparse
import json
import time
from pathlib import Path

import numpy as np

from ai_lab.metrics import accuracy, confusion_matrix_binary, mae, mse


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    test_n = int(round(n * test_ratio))
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def fit_linear_least_squares(x: np.ndarray, y: np.ndarray):
    # x: (n,)  -> design matrix [1, x]
    X = np.stack([np.ones_like(x), x], axis=1)  # (n, 2)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)  # [bias, slope]
    return w


def predict_linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w[0] + w[1] * x


def fit_nearest_centroid(X: np.ndarray, y: np.ndarray):
    c0 = X[y == 0].mean(axis=0)
    c1 = X[y == 1].mean(axis=0)
    return c0, c1


def predict_nearest_centroid(X: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    d0 = np.sum((X - c0) ** 2, axis=1)
    d1 = np.sum((X - c1) ** 2, axis=1)
    return (d1 < d0).astype(np.int64)


def save_report(out_dir: str, report: dict) -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_dir) / f"split_metrics_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["regression", "classification"], default="regression")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--out-dir", type=str, default="outputs")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    report = {"task": args.task, "config": vars(args), "train": {}, "test": {}}

    if args.task == "regression":
        # y = 3x + 2 + noise
        x = rng.uniform(-2.0, 2.0, size=(args.n,))
        y = 3.0 * x + 2.0 + rng.normal(0.0, args.noise, size=(args.n,))

        Xtr, Xte, ytr, yte = train_test_split(x, y, test_ratio=args.test_ratio, seed=args.seed)

        w = fit_linear_least_squares(Xtr, ytr)
        yhat_tr = predict_linear(Xtr, w)
        yhat_te = predict_linear(Xte, w)

        report["model"] = {
            "type": "linear_least_squares",
            "bias": float(w[0]),
            "slope": float(w[1]),
        }
        report["train"] = {"mse": mse(yhat_tr, ytr), "mae": mae(yhat_tr, ytr)}
        report["test"] = {"mse": mse(yhat_te, yte), "mae": mae(yhat_te, yte)}

        print("REGRESSION")
        print(f"fit: y = {w[1]:.3f} * x + {w[0]:.3f}")
        print(f"train: MSE={report['train']['mse']:.4f}  MAE={report['train']['mae']:.4f}")
        print(f"test : MSE={report['test']['mse']:.4f}  MAE={report['test']['mae']:.4f}")

    else:
        # 2D points, label by linear boundary + noise
        X = rng.normal(0.0, 1.0, size=(args.n, 2))
        true_w = np.array([1.2, -0.8])
        logits = X @ true_w + rng.normal(0.0, args.noise, size=(args.n,))
        y = (logits > 0.0).astype(np.int64)

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_ratio=args.test_ratio, seed=args.seed)

        c0, c1 = fit_nearest_centroid(Xtr, ytr)
        yhat_tr = predict_nearest_centroid(Xtr, c0, c1)
        yhat_te = predict_nearest_centroid(Xte, c0, c1)

        report["model"] = {"type": "nearest_centroid"}
        report["train"] = {
            "accuracy": accuracy(yhat_tr, ytr),
            "confusion": confusion_matrix_binary(yhat_tr, ytr),
        }
        report["test"] = {
            "accuracy": accuracy(yhat_te, yte),
            "confusion": confusion_matrix_binary(yhat_te, yte),
        }

        print("CLASSIFICATION")
        print(f"train: acc={report['train']['accuracy']:.4f}  cm={report['train']['confusion']}")
        print(f"test : acc={report['test']['accuracy']:.4f}  cm={report['test']['confusion']}")

    run_dir = save_report(args.out_dir, report)
    print(f"\nSaved report to: {run_dir}/report.json")


if __name__ == "__main__":
    main()
