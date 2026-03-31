import argparse
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class CNN1DBaseline(nn.Module):
    def __init__(self, n_features: int, horizon: int = 12):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B,F,L)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.gap(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


@dataclass
class RunOutputs:
    results: Dict[str, float]
    meta_test: List[Tuple[str, str, pd.Timestamp]]
    y_true: np.ndarray
    y_pred: np.ndarray


def _bad(arr: np.ndarray) -> Dict[str, int]:
    return {"nan": int(np.isnan(arr).sum()), "inf": int(np.isinf(arr).sum())}


class EarlyStopping:
    def __init__(self, patience: int = 12, min_delta: float = 1e-4, min_epochs: int = 20, restore_best: bool = True):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_epochs = int(min_epochs)
        self.restore_best = bool(restore_best)

        self.best = float("inf")
        self.best_state = None
        self.bad_epochs = 0

    def step(self, epoch: int, val_loss: float, model: nn.Module) -> bool:
        improved = val_loss < (self.best - self.min_delta)

        if improved:
            self.best = float(val_loss)
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.bad_epochs = 0
        else:
            if epoch >= self.min_epochs:
                self.bad_epochs += 1

        return (epoch >= self.min_epochs) and (self.bad_epochs >= self.patience)

    def restore(self, model: nn.Module) -> None:
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


def baseline_persistence(X: np.ndarray, feature_cols: List[str], target_col: str, horizon: int) -> np.ndarray:
    if target_col not in feature_cols:
        raise ValueError("Persistence baseline requires target_col inside feature_cols.")
    idx = feature_cols.index(target_col)
    last = X[:, -1, idx]
    return np.repeat(last[:, None], horizon, axis=1)


def baseline_seasonal_naive(X: np.ndarray, feature_cols: List[str], target_col: str, horizon: int, history_len: int):
    if history_len < 12:
        return None
    if target_col not in feature_cols:
        return None
    idx = feature_cols.index(target_col)
    last12 = X[:, -12:, idx]
    if last12.shape[1] != horizon:
        return None
    return last12.copy()


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    station_col: str,
    date_col: str,
    dept_col: str,
    dept_split: Dict[str, List[str]],
    cuts: Dict[str, pd.Timestamp],
    history_len: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    min_delta: float,
    min_epochs: int,
    disable_early_stopping: bool = False,
    device: Optional[str] = None,
) -> RunOutputs:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_depts = set(dept_split["train"])
    val_depts = set(dept_split.get("val", []))
    test_depts = set(dept_split["test"])

    df_train_dept = df[df[dept_col].isin(train_depts)].copy()
    df_val_dept = df[df[dept_col].isin(val_depts)].copy() if len(val_depts) else None
    df_test_dept = df[df[dept_col].isin(test_depts)].copy()

    scaler = fit_scaler_on_train(df_train_dept, feature_cols)
    df_train_s = apply_scaler(df_train_dept, feature_cols, scaler)
    df_test_s = apply_scaler(df_test_dept, feature_cols, scaler)
    df_val_s = apply_scaler(df_val_dept, feature_cols, scaler) if df_val_dept is not None else None

    train_end = cuts["train_end"]
    val_end = cuts.get("val_end", None)

    def plus_month(ts: pd.Timestamp) -> pd.Timestamp:
        return (ts + pd.offsets.MonthBegin(1)).normalize()

    val_start = plus_month(train_end)
    test_start = plus_month(val_end) if val_end is not None else val_start

    group_cols = [dept_col, station_col]

    X_train, y_train, _ = make_sequences_strict(
        df_train_s, history_len, horizon, group_cols, date_col, feature_cols, target_col,
        y_end_max=train_end, y_start_min=None
    )

    X_val = y_val = None
    if df_val_s is not None and not df_val_s.empty and val_end is not None:
        X_val, y_val, _ = make_sequences_strict(
            df_val_s, history_len, horizon, group_cols, date_col, feature_cols, target_col,
            y_end_max=val_end, y_start_min=val_start
        )

    X_test, y_test, meta_test = make_sequences_strict(
        df_test_s, history_len, horizon, group_cols, date_col, feature_cols, target_col,
        y_end_max=None, y_start_min=test_start
    )

    print("\n=== Sanity checks ===")
    print("X_train:", X_train.shape, _bad(X_train), "| y_train:", y_train.shape, _bad(y_train))
    if X_val is not None:
        print("X_val  :", X_val.shape, _bad(X_val), "| y_val  :", y_val.shape, _bad(y_val))
    print("X_test :", X_test.shape, _bad(X_test), "| y_test :", y_test.shape, _bad(y_test))
    print("feature_cols:", feature_cols)

    pers_pred = baseline_persistence(X_test, feature_cols, target_col, horizon)
    pers_res = {"MAE": mae(y_test, pers_pred), "RMSE": rmse(y_test, pers_pred), "sMAPE(%)": smape(y_test, pers_pred)}

    seas_pred = baseline_seasonal_naive(X_test, feature_cols, target_col, horizon, history_len)
    seas_res = None
    if seas_pred is not None:
        seas_res = {"MAE": mae(y_test, seas_pred), "RMSE": rmse(y_test, seas_pred), "sMAPE(%)": smape(y_test, seas_pred)}

    print("\n=== Baselines (on TEST windows, scaled) ===")
    print("Persistence  :", pers_res)
    if seas_res is not None:
        print("SeasonalNaive:", seas_res)
    else:
        print("SeasonalNaive: N/A (needs history_len>=12 and target in features)")

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=batch_size, shuffle=False) if X_val is not None else None
    test_loader = DataLoader(WindowDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    model = CNN1DBaseline(n_features=len(feature_cols), horizon=horizon).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=max(2, patience // 3), threshold=min_delta
    )
    stopper = EarlyStopping(patience=patience, min_delta=min_delta, min_epochs=min_epochs, restore_best=True)

    def run_epoch(loader: DataLoader, train: bool) -> float:
        model.train(train)
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if train:
                optim.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            losses.append(loss.detach().cpu().item())
        return float(np.mean(losses)) if losses else float("inf")

    for ep in range(1, epochs + 1):
        tr_loss = run_epoch(train_loader, train=True)
        va_loss = run_epoch(val_loader, train=False) if val_loader is not None else tr_loss

        scheduler.step(va_loss)
        lr_now = optim.param_groups[0]["lr"]

        print(f"Epoch {ep:03d}/{epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | lr={lr_now:.2e}")

        if (not disable_early_stopping) and (val_loader is not None) and stopper.step(ep, va_loss, model):
            print(f"EarlyStopping at epoch {ep} (best val_loss={stopper.best:.4f})")
            break

    if not disable_early_stopping:
        stopper.restore(model)

    model.eval()
    yp, yt = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb.to(device)).cpu().numpy()
            yp.append(pred)
            yt.append(yb.numpy())
    y_pred = np.concatenate(yp, axis=0)
    y_true = np.concatenate(yt, axis=0)

    results = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE(%)": smape(y_true, y_pred),
        "n_train_seq": int(X_train.shape[0]),
        "n_test_seq": int(X_test.shape[0]),
        "n_features": int(len(feature_cols)),
        "history_len": int(history_len),
        "horizon": int(horizon),
        "device": device,
        "baseline_persistence_MAE": pers_res["MAE"],
        "baseline_persistence_RMSE": pers_res["RMSE"],
    }
    if seas_res is not None:
        results["baseline_seasonal_MAE"] = seas_res["MAE"]
        results["baseline_seasonal_RMSE"] = seas_res["RMSE"]

    return RunOutputs(results=results, meta_test=meta_test, y_true=y_true, y_pred=y_pred)


def plot_examples(meta_test, y_true, y_pred, n_examples=2):
    if not meta_test:
        return
    idxs = list(range(len(meta_test)))
    rng = np.random.default_rng(42)
    rng.shuffle(idxs)
    idxs = idxs[:n_examples]

    plt.figure(figsize=(12, 4 * len(idxs)))
    for k, i in enumerate(idxs, start=1):
        dept, st, start = meta_test[i]
        x = np.arange(1, y_true.shape[1] + 1)
        plt.subplot(len(idxs), 1, k)
        plt.plot(x, y_true[i], label="True")
        plt.plot(x, y_pred[i], label="Pred")
        plt.title(f"Dept={dept} | Station={st} | start={pd.to_datetime(start).date()} | horizon={y_true.shape[1]}")
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.tight_layout()
    plt.show()


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--train_depts", type=str, required=True, help="ex: 01,38,73")
    ap.add_argument("--val_depts", type=str, default="", help="ex: 74")
    ap.add_argument("--test_depts", type=str, required=True, help="ex: 69")
    ap.add_argument("--target", type=str, default="TM")
    ap.add_argument("--features", type=str, default="")
    ap.add_argument("--history_len", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--train_end", type=str, required=True, help="YYYY-MM-01")
    ap.add_argument("--val_end", type=str, default="", help="YYYY-MM-01 (optional)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--min_epochs", type=int, default=30)
    ap.add_argument("--disable_early_stopping", action="store_true", help="If set, run all epochs (no early stop).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_months_per_station", type=int, default=24)
    ap.add_argument("--plot_examples", type=int, default=1)
    args = ap.parse_args()

    if args.history_len < 1:
        raise ValueError("history_len must be >= 1")


    set_seed(args.seed)

    train_depts = _parse_list(args.train_depts)
    val_depts = _parse_list(args.val_depts) if args.val_depts.strip() else []
    test_depts = _parse_list(args.test_depts)

    feat_list = [c.strip() for c in args.features.split(",") if c.strip()] if args.features.strip() else None

    df, target_col, feature_cols, station_col, date_col, dept_col = load_and_preprocess_multi_dept(
        data_dir=args.data_dir,
        dept_codes=list(dict.fromkeys(train_depts + val_depts + test_depts)),
        target=args.target,
        features=feat_list,
        min_months_per_station=args.min_months_per_station,
        add_month_features=True,
        include_target_as_feature=True,
        verbose=True,
    )

    dept_split = split_by_department(df, dept_col, train_depts, val_depts, test_depts)
    cuts = split_by_time_strict(df, date_col, train_end=args.train_end, val_end=(args.val_end if args.val_end.strip() else None))

    print("\n=== Department split ===")
    print("Train depts:", dept_split["train"])
    print("Val depts  :", dept_split["val"])
    print("Test depts :", dept_split["test"])
    print("Cuts:", {k: str(v.date()) for k, v in cuts.items()})

    out = train_and_evaluate(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        station_col=station_col,
        date_col=date_col,
        dept_col=dept_col,
        dept_split=dept_split,
        cuts=cuts,
        history_len=args.history_len,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        min_delta=args.min_delta,
        min_epochs=args.min_epochs,
        disable_early_stopping=args.disable_early_stopping,
    )

    print("\n=== Test results (unseen departments) ===")
    print(pd.DataFrame([out.results]).to_string(index=False))

    if args.plot_examples == 1:
        plot_examples(out.meta_test, out.y_true, out.y_pred, n_examples=2)


if __name__ == "__main__":
    main()
