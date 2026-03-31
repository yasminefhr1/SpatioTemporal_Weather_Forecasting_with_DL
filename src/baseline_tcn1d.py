import argparse
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.data_processing import (
    load_and_preprocess_multi_dept,
    split_by_department,
    split_by_time_strict,
    make_sequences_strict,
    fit_scaler_on_train,
    apply_scaler,
)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()  # (N, L, F)
        self.y = torch.from_numpy(y).float()  # (N, H)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        # Conv1d expects (C, L)
        x = self.X[idx].transpose(0, 1)  # (F, L)
        y = self.y[idx]                  # (H,)
        return x, y


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


class Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class ResidualTCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        pad = (k - 1) * dilation  # causal padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=dilation, padding=pad)
        self.chomp1 = Chomp1d(pad)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=dilation, padding=pad)
        self.chomp2 = Chomp1d(pad)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.final_act = nn.ReLU()

    def forward(self, x):
        y = self.drop1(self.act1(self.chomp1(self.conv1(x))))
        y = self.drop2(self.act2(self.chomp2(self.conv2(y))))
        return self.final_act(y + self.down(x))


class TCNForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            layers.append(ResidualTCNBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, horizon),
        )

    def forward(self, x):
        z = self.tcn(x)                  # (N, C, L)
        z = self.pool(z).squeeze(-1)     # (N, C)
        return self.head(z)              # (N, H)


@dataclass
class TrainCfg:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    patience: int
    min_epochs: int
    min_delta: float
    disable_early_stopping: bool


class EarlyStopper:
    def __init__(self, patience: int, min_epochs: int, min_delta: float):
        self.patience = patience
        self.min_epochs = min_epochs
        self.min_delta = min_delta
        self.best = np.inf
        self.bad = 0
        self.best_state = None

    def step(self, epoch: int, val_loss: float, model: nn.Module) -> bool:
        if val_loss + self.min_delta < self.best:
            self.best = val_loss
            self.bad = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        if epoch >= self.min_epochs:
            self.bad += 1
        return (epoch >= self.min_epochs) and (self.bad >= self.patience)

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


def train_one(model, loader, optimizer, loss_fn, device, grad_clip: float) -> float:
    model.train()
    losses = []
    for batch in loader:
        xb, yb = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_one(model, loader, loss_fn, device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        xb, yb = to_device(batch, device)
        pred = model(xb)
        losses.append(loss_fn(pred, yb).item())
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        xb, _ = to_device(batch, device)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--train_depts", type=str, required=True)
    ap.add_argument("--val_depts", type=str, required=True)
    ap.add_argument("--test_depts", type=str, required=True)
    ap.add_argument("--train_end", type=str, required=True)
    ap.add_argument("--val_end", type=str, default=None)

    ap.add_argument("--target", type=str, default="TM")
    ap.add_argument("--history_len", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--min_months_per_station", type=int, default=24)

    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min_epochs", type=int, default=30)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--disable_early_stopping", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--channels", type=str, default="64,64,64")
    ap.add_argument("--kernel_size", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.15)

    args = ap.parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_depts = parse_list(args.train_depts)
    val_depts = parse_list(args.val_depts)
    test_depts = parse_list(args.test_depts)

    df, target_col, feature_cols, station_col, date_col, dept_col = load_and_preprocess_multi_dept(
        data_dir=args.data_dir,
        dept_codes=sorted(set(train_depts + val_depts + test_depts)),
        target=args.target,
        min_months_per_station=args.min_months_per_station,
        add_month_features=True,
        include_target_as_feature=True,
        verbose=True,
    )

    dept_split = split_by_department(df, dept_col, train_depts, val_depts, test_depts)
    cuts = split_by_time_strict(df, date_col, train_end=args.train_end, val_end=args.val_end)

    print("\n=== Department split ===")
    print("Train depts:", dept_split["train"])
    print("Val depts  :", dept_split["val"])
    print("Test depts :", dept_split["test"])
    print("Cuts:", {k: str(v.date()) for k, v in cuts.items()})

    train_mask = df[dept_col].isin(dept_split["train"]) & (pd.to_datetime(df[date_col]) <= cuts["train_end"])
    scaler = fit_scaler_on_train(df.loc[train_mask].copy(), feature_cols)
    df_scaled = apply_scaler(df, feature_cols, scaler)

    # windows
    X_train, y_train, _ = make_sequences_strict(
        df_scaled[df_scaled[dept_col].isin(dept_split["train"])],
        history_len=args.history_len,
        horizon=args.horizon,
        group_cols=[dept_col, station_col],
        time_col=date_col,
        feature_cols=feature_cols,
        target_col=target_col,
        y_end_max=cuts["train_end"],
        y_start_min=None,
    )

    y_start_min = cuts["train_end"] + pd.offsets.MonthBegin(1)
    y_end_max = cuts.get("val_end", None)
    X_val, y_val, _ = make_sequences_strict(
        df_scaled[df_scaled[dept_col].isin(dept_split["val"])],
        history_len=args.history_len,
        horizon=args.horizon,
        group_cols=[dept_col, station_col],
        time_col=date_col,
        feature_cols=feature_cols,
        target_col=target_col,
        y_end_max=y_end_max,
        y_start_min=y_start_min,
    )

    test_start = (cuts["val_end"] + pd.offsets.MonthBegin(1)) if ("val_end" in cuts) else y_start_min
    X_test, y_test, meta_test = make_sequences_strict(
        df_scaled[df_scaled[dept_col].isin(dept_split["test"])],
        history_len=args.history_len,
        horizon=args.horizon,
        group_cols=[dept_col, station_col],
        time_col=date_col,
        feature_cols=feature_cols,
        target_col=target_col,
        y_end_max=None,
        y_start_min=test_start,
    )

    print("\n=== Sanity checks ===")
    print("X_train:", X_train.shape, "| y_train:", y_train.shape)
    print("X_val  :", X_val.shape, "| y_val  :", y_val.shape)
    print("X_test :", X_test.shape, "| y_test :", y_test.shape)
    print("feature_cols:", feature_cols)

    # loaders
    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False) if len(X_val) else None
    test_loader = DataLoader(WindowDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    channels = [int(x) for x in parse_list(args.channels)]
    model = TCNForecaster(
        n_features=len(feature_cols),
        horizon=args.horizon,
        channels=channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    stopper = EarlyStopper(args.patience, args.min_epochs, args.min_delta)

    print("\n=== Training TCN ===")
    for ep in range(1, args.epochs + 1):
        tr = train_one(model, train_loader, optimizer, loss_fn, device, args.grad_clip)
        if val_loader is not None:
            va = eval_one(model, val_loader, loss_fn, device)
            scheduler.step(va)
        else:
            va = float("nan")
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {ep:03d}/{args.epochs} | train_loss={tr:.4f} | val_loss={va:.4f} | lr={lr:.2e}")

        if (not args.disable_early_stopping) and (val_loader is not None) and stopper.step(ep, va, model):
            print(f"EarlyStopping at epoch {ep} (best val_loss={stopper.best:.4f})")
            break

    if (not args.disable_early_stopping) and (val_loader is not None):
        stopper.restore(model)

    y_pred = predict(model, test_loader, device)
    out = {"MAE": mae(y_test, y_pred), "RMSE": rmse(y_test, y_pred), "sMAPE(%)": smape(y_test, y_pred)}
    print("\n=== Test results (unseen departments) ===")
    print(out)

    # plot 2 examples
    try:
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(len(y_test), size=min(2, len(y_test)), replace=False)
        fig, axes = plt.subplots(len(idxs), 1, figsize=(10, 4 * len(idxs)))
        if len(idxs) == 1:
            axes = [axes]
        for ax, i in zip(axes, idxs):
            dept, st, start = meta_test[i]
            ax.plot(np.arange(1, args.horizon + 1), y_test[i], label="True")
            ax.plot(np.arange(1, args.horizon + 1), y_pred[i], label="Pred")
            ax.set_title(f"Dept={dept} | Station={st} | start={pd.to_datetime(start).date()} | horizon={args.horizon}")
            ax.set_xlabel("Forecast step (month)")
            ax.set_ylabel("Target (scaled)")
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("[plot] skipped:", e)


if __name__ == "__main__":
    main()
