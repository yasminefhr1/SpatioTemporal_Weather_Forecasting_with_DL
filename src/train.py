import argparse
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

from src.models import PatchTST, Seq2SeqAttentionModel
from src.baseline_tcn1d import TCNForecaster
from src.baselines_cnn import CNN1DBaseline


# -----------------------
# Utils
# -----------------------
def parse_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_str: Optional[str] = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE(%)": smape(y_true, y_pred),
        "MAPE(%)": mape(y_true, y_pred),
    }

def metrics_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    H = y_true.shape[1]
    out = {}
    for h in range(H):
        out[f"t+{h+1}"] = metrics_dict(y_true[:, [h]], y_pred[:, [h]])
    return out


class WindowDataset(Dataset):
    """X: (N, L, F) | y: (N, H)"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class EarlyStopper:
    def __init__(self, patience: int = 10, min_epochs: int = 15, min_delta: float = 1e-4):
        self.patience = int(patience)
        self.min_epochs = int(min_epochs)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.bad = 0
        self.best_state = None

    def step(self, epoch: int, val_loss: float, model: nn.Module) -> bool:
        improved = val_loss < (self.best - self.min_delta)
        if improved:
            self.best = float(val_loss)
            self.bad = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        if epoch >= self.min_epochs:
            self.bad += 1
        return (epoch >= self.min_epochs) and (self.bad >= self.patience)

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# -----------------------
# Config
# -----------------------
@dataclass
class TrainConfig:
    model: str = "patchtst"  # patchtst|seq2seq|tcn|cnn

    data_dir: str = "data/raw"
    train_depts: str = "01,38,73"
    val_depts: str = "74"
    test_depts: str = "69"
    train_end: str = "2018-12-01"
    val_end: Optional[str] = "2020-12-01"
    target: str = "TM"
    min_months_per_station: int = 24

    history_len: int = 24
    horizon: int = 12

    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 10
    min_epochs: int = 15
    min_delta: float = 1e-4
    disable_early_stopping: bool = False

    # seq2seq
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    teacher_forcing_ratio: float = 0.5
    teacher_forcing_decay: float = 0.98

    # patchtst
    patch_len: int = 4
    d_model: int = 64
    n_heads: int = 4

    # tcn
    channels: str = "64,64,64"
    kernel_size: int = 3

    seed: int = 42
    device: Optional[str] = None


    # plotting
    plot_examples: int = 2  # number of True vs Pred plots after training (0 to disable)
    def to_dict(self) -> Dict:
        return asdict(self)


# -----------------------
# Data prep (shared)
# -----------------------
def prepare_data(cfg: TrainConfig):
    train_depts = parse_list(cfg.train_depts)
    val_depts = parse_list(cfg.val_depts)
    test_depts = parse_list(cfg.test_depts)

    df, target_col, feature_cols, station_col, date_col, dept_col = load_and_preprocess_multi_dept(
        data_dir=cfg.data_dir,
        dept_codes=sorted(set(train_depts + val_depts + test_depts)),
        target=cfg.target,
        min_months_per_station=cfg.min_months_per_station,
        add_month_features=True,
        include_target_as_feature=True,  # needed for persistence baseline
        verbose=True,
    )

    dept_split = split_by_department(df, dept_col, train_depts, val_depts, test_depts)
    cuts = split_by_time_strict(df, date_col, train_end=cfg.train_end, val_end=cfg.val_end)

    # fit scaler ONLY on training partition (train depts + time <= train_end)
    train_mask = df[dept_col].isin(dept_split["train"]) & (pd.to_datetime(df[date_col]) <= cuts["train_end"])
    scaler = fit_scaler_on_train(df.loc[train_mask].copy(), feature_cols)
    df_scaled = apply_scaler(df, feature_cols, scaler)

    group_cols = [dept_col, station_col]

    X_train, y_train, _ = make_sequences_strict(
        df_scaled[df_scaled[dept_col].isin(dept_split["train"])],
        history_len=cfg.history_len,
        horizon=cfg.horizon,
        group_cols=group_cols,
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
        history_len=cfg.history_len,
        horizon=cfg.horizon,
        group_cols=group_cols,
        time_col=date_col,
        feature_cols=feature_cols,
        target_col=target_col,
        y_end_max=y_end_max,
        y_start_min=y_start_min,
    )

    test_start = (cuts["val_end"] + pd.offsets.MonthBegin(1)) if (cuts.get("val_end", None) is not None) else y_start_min
    X_test, y_test, meta_test = make_sequences_strict(
        df_scaled[df_scaled[dept_col].isin(dept_split["test"])],
        history_len=cfg.history_len,
        horizon=cfg.horizon,
        group_cols=group_cols,
        time_col=date_col,
        feature_cols=feature_cols,
        target_col=target_col,
        y_end_max=None,
        y_start_min=test_start,
    )

    loaders = {
        "train": DataLoader(WindowDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True),
        "val": DataLoader(WindowDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False) if len(X_val) else None,
        "test": DataLoader(WindowDataset(X_test, y_test), batch_size=cfg.batch_size, shuffle=False),
    }

    meta = {
        "target_col": target_col,
        "feature_cols": feature_cols,
        "station_col": station_col,
        "date_col": date_col,
        "dept_col": dept_col,
        "dept_split": dept_split,
        "cuts": {k: str(v) if v is not None else None for k, v in cuts.items()},
        "meta_test": [(str(a), str(b), str(c)) for (a, b, c) in meta_test],
    }

    return loaders, (X_test, y_test), meta


# -----------------------
# Model factory
# -----------------------
def build_model(cfg: TrainConfig, n_features: int) -> nn.Module:
    m = cfg.model.lower().strip()

    if m == "patchtst":
        if cfg.history_len < cfg.patch_len:
            raise ValueError(f"history_len ({cfg.history_len}) must be >= patch_len ({cfg.patch_len})")
        return PatchTST(
            n_features=n_features,
            history_len=cfg.history_len,
            horizon=cfg.horizon,
            patch_len=cfg.patch_len,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_revin=True,
            channel_independence=True,
        )

    if m in ("seq2seq", "seq2seq_attention"):
        return Seq2SeqAttentionModel(
            input_size=n_features,
            hidden_size=cfg.hidden_size,
            horizon=cfg.horizon,
            enc_layers=cfg.num_layers,
            dec_layers=cfg.num_layers,
            dropout=cfg.dropout,
            attention_type="bahdanau",
            bidirectional_encoder=True,
        )

    if m in ("tcn", "tcn1d"):
        channels = [int(x) for x in parse_list(cfg.channels)]
        return TCNForecaster(
            n_features=n_features,
            horizon=cfg.horizon,
            channels=channels,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )

    if m in ("cnn", "cnn1d"):
        return CNN1DBaseline(n_features=n_features, horizon=cfg.horizon)

    raise ValueError(f"Unknown --model '{cfg.model}'. Use patchtst|seq2seq|tcn|cnn.")


def forward_model(cfg: TrainConfig, model: nn.Module, X: torch.Tensor, y: Optional[torch.Tensor], epoch: int) -> torch.Tensor:
    m = cfg.model.lower().strip()

    if m in ("seq2seq", "seq2seq_attention"):
        tf_ratio = cfg.teacher_forcing_ratio * (cfg.teacher_forcing_decay ** epoch)
        return model(X, tgt=y, teacher_forcing_ratio=tf_ratio)

    if m in ("tcn", "tcn1d"):
        return model(X.transpose(1, 2))

    if m in ("cnn", "cnn1d"):
        return model(X)

    return model(X)


@torch.no_grad()
def predict(cfg: TrainConfig, model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for X, _ in loader:
        X = X.to(device)
        yhat = forward_model(cfg, model, X, None, epoch=10**9)
        preds.append(yhat.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_one_epoch(cfg: TrainConfig, model: nn.Module, loader: DataLoader, opt, loss_fn, device: torch.device, epoch: int) -> float:
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        yhat = forward_model(cfg, model, X, y, epoch)
        loss = loss_fn(yhat, y)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        bs = X.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def eval_loss(cfg: TrainConfig, model: nn.Module, loader: DataLoader, loss_fn, device: torch.device, epoch: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        yhat = forward_model(cfg, model, X, None, epoch)
        loss = loss_fn(yhat, y)
        bs = X.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


def run_training(cfg: TrainConfig) -> Dict:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    print(f"\n🔧 Device: {device}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{cfg.model}_{ts}"
    run_dir = Path("results") / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    loaders, (X_test_np, y_test_np), meta = prepare_data(cfg)
    feature_cols = meta["feature_cols"]
    n_features = len(feature_cols)

    with open(run_dir / "data_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Save test arrays for evaluate.py (baseline comparison)
    np.savez_compressed(run_dir / "x_test.npz", X_test=X_test_np)
    np.savez_compressed(run_dir / "y_test.npz", y_test=y_test_np)

    model = build_model(cfg, n_features=n_features).to(device)
    print(f"🏗️  Model={cfg.model} | n_features={n_features} | history_len={cfg.history_len} | horizon={cfg.horizon}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    stopper = EarlyStopper(patience=cfg.patience, min_epochs=cfg.min_epochs, min_delta=cfg.min_delta)

    log_rows = []
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        tr = train_one_epoch(cfg, model, loaders["train"], opt, loss_fn, device, epoch)

        if loaders["val"] is not None:
            va = eval_loss(cfg, model, loaders["val"], loss_fn, device, epoch)
        else:
            va = tr

        lr_now = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | train_loss={tr:.4f} | val_loss={va:.4f} | lr={lr_now:.2e}")
        log_rows.append({"epoch": epoch + 1, "train_loss": tr, "val_loss": va, "lr": lr_now})

        if va < best_val:
            best_val = va
            torch.save({"model_state": model.state_dict(), "cfg": cfg.to_dict()}, run_dir / "best_model.pt")
            print("  ✓ best_model.pt updated")

        if (not cfg.disable_early_stopping) and (loaders["val"] is not None):
            if stopper.step(epoch + 1, va, model):
                print(f"⏹️  Early stopping (best val={stopper.best:.4f})")
                break

    stopper.restore_best(model)

    torch.save({"model_state": model.state_dict(), "cfg": cfg.to_dict()}, run_dir / "final_model.pt")
    pd.DataFrame(log_rows).to_csv(run_dir / "training_log.csv", index=False)

    # Test predictions + metrics
    y_pred = predict(cfg, model, loaders["test"], device=device)
    y_true = y_test_np

    test_global = metrics_dict(y_true, y_pred)
    test_by_h = metrics_by_horizon(y_true, y_pred)

    np.savez_compressed(run_dir / "test_predictions.npz", y_true=y_true, y_pred=y_pred)
    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"global": test_global, "by_horizon": test_by_h}, f, indent=2)

    print("\n📌 TEST metrics (global):", test_global)
    print("✅ Run saved in:", str(run_dir))

    # -----------------------
    # Plot True vs Pred examples (like your old figure)
    # -----------------------
    try:
        n_plots = int(getattr(cfg, "plot_examples", 2))
        if n_plots > 0:
            import matplotlib.pyplot as plt

            meta_test = meta.get("meta_test", [])
            if len(meta_test) == 0:
                meta_test = [("?", "?", "?")] * len(y_true)

            rng = np.random.default_rng(cfg.seed)
            idxs = rng.choice(len(y_true), size=min(n_plots, len(y_true)), replace=False)

            fig, axes = plt.subplots(len(idxs), 1, figsize=(12, 4 * len(idxs)))
            if len(idxs) == 1:
                axes = [axes]

            for ax, i in zip(axes, idxs):
                dept, st, start = meta_test[i]
                ax.plot(np.arange(1, cfg.horizon + 1), y_true[i], label="True")
                ax.plot(np.arange(1, cfg.horizon + 1), y_pred[i], label="Pred")
                try:
                    start_str = pd.to_datetime(start).date()
                except Exception:
                    start_str = start
                ax.set_title(f"Dept={dept} | Station={st} | start={start_str} | horizon={cfg.horizon}")
                ax.set_xlabel("Forecast step (month)")
                ax.set_ylabel("Target (scaled)")
                ax.grid(True, alpha=0.3)
                ax.legend()

            plt.tight_layout()
            out_png = run_dir / "true_vs_pred_examples.png"
            plt.savefig(out_png, dpi=150)
            plt.show()
            print(f"[plot] saved: {out_png}")
    except Exception as e:
        print("[plot] skipped:", e)

    return {"run_dir": str(run_dir), "best_val_loss": float(best_val), "test_metrics": test_global}


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--model", type=str, default="patchtst", help="patchtst|seq2seq|tcn|cnn")

    # data
    p.add_argument("--data_dir", type=str, default="data/raw")
    p.add_argument("--train_depts", type=str, default="01,38,73")
    p.add_argument("--val_depts", type=str, default="74")
    p.add_argument("--test_depts", type=str, default="69")
    p.add_argument("--train_end", type=str, default="2018-12-01")
    p.add_argument("--val_end", type=str, default="2020-12-01")
    p.add_argument("--target", type=str, default="TM")
    p.add_argument("--min_months_per_station", type=int, default=24)

    # windows
    p.add_argument("--history_len", type=int, default=24)
    p.add_argument("--horizon", type=int, default=12)

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_epochs", type=int, default=15)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--disable_early_stopping", action="store_true")

    # seq2seq
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    p.add_argument("--teacher_forcing_decay", type=float, default=0.98)

    # patchtst
    p.add_argument("--patch_len", type=int, default=4)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)

    # tcn
    p.add_argument("--channels", type=str, default="64,64,64")
    p.add_argument("--kernel_size", type=int, default=3)

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--plot_examples", type=int, default=2, help="Number of True vs Pred plots after training (0 to disable)")
    args = p.parse_args()
    return TrainConfig(**vars(args))


def main():
    cfg = parse_args()
    run_training(cfg)


if __name__ == "__main__":
    main()
