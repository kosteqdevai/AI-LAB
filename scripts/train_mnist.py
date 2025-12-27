import csv
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torchvision import datasets, transforms


@dataclass
class Cfg:
    seed: int = 123
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 2
    out_dir: str = "outputs"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "no-git"


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = Cfg(**yaml.safe_load(f))

    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.out_dir) / f"mnist_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # zapis śladów
    (run_dir / "commit.txt").write_text(get_git_commit() + "\n", encoding="utf-8")
    (run_dir / "device.txt").write_text(device + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(
        Path(cfg_path).read_text(encoding="utf-8"), encoding="utf-8"
    )

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2
    )

    model = SimpleNet().to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            tl, ta, n = 0.0, 0.0, 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

                bs = x.size(0)
                tl += loss.item() * bs
                ta += accuracy(logits, y) * bs
                n += bs

            train_loss = tl / n
            train_acc = ta / n

            model.eval()
            vl, va, m = 0.0, 0.0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = loss_fn(logits, y)

                    bs = x.size(0)
                    vl += loss.item() * bs
                    va += accuracy(logits, y) * bs
                    m += bs

            test_loss = vl / m
            test_acc = va / m

            writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
            f.flush()

            print(
                f"epoch {epoch} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"test loss {test_loss:.4f} acc {test_acc:.4f}"
            )

    # checkpoint
    torch.save(model.state_dict(), run_dir / "model.pt")
    print(f"\nSaved run to: {run_dir}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    args = p.parse_args()
    main(args.config)
