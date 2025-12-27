import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def make_run_dir(base: str) -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base) / f"linreg_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def mse(y_pred: np.ndarray, y: np.ndarray) -> float:
    err = y_pred - y
    return float(np.mean(err * err))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--out-dir", type=str, default="outputs")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # Synthetic data: y = 3x + 2 + noise
    x = rng.uniform(-2.0, 2.0, size=(args.n, 1))
    y = 3.0 * x[:, 0] + 2.0 + rng.normal(0.0, args.noise, size=(args.n,))

    # Design matrix with bias term
    X = np.concatenate([np.ones((args.n, 1)), x], axis=1)  # (n, 2)
    w = rng.normal(0.0, 0.1, size=(2,))  # [bias, slope]

    run_dir = make_run_dir(args.out_dir)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "bias", "slope"])

        for step in range(1, args.steps + 1):
            y_pred = X @ w
            loss = mse(y_pred, y)

            # Gradient of MSE wrt w: (2/n) * X^T (Xw - y)
            grad = (2.0 / args.n) * (X.T @ (y_pred - y))
            w -= args.lr * grad

            if step % args.log_every == 0 or step == 1 or step == args.steps:
                writer.writerow([step, loss, float(w[0]), float(w[1])])
                f.flush()
                print(f"step {step:4d} | loss {loss:.6f} | bias {w[0]:.3f} | slope {w[1]:.3f}")

    (run_dir / "weights.txt").write_text(f"bias={w[0]}\nslope={w[1]}\n", encoding="utf-8")
    print(f"\nSaved run to: {run_dir}")


if __name__ == "__main__":
    main()
