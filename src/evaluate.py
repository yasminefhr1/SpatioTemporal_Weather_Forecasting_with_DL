#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# -----------------------
# Metrics
# -----------------------
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
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "sMAPE(%)": smape(y_true, y_pred), "MAPE(%)": mape(y_true, y_pred)}

def metrics_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    H = y_true.shape[1]
    out = {}
    for h in range(H):
        out[f"t+{h+1}"] = metrics_dict(y_true[:, [h]], y_pred[:, [h]])
    return out


# -----------------------
# Baselines
# -----------------------
def baseline_persistence(X: np.ndarray, feature_cols: List[str], target_col: str, horizon: int) -> np.ndarray:
    idx = feature_cols.index(target_col)
    last = X[:, -1, idx]
    return np.repeat(last[:, None], horizon, axis=1)

def baseline_seasonal_naive(X: np.ndarray, feature_cols: List[str], target_col: str, horizon: int, history_len: int) -> Optional[np.ndarray]:
    if history_len < 12:
        return None
    idx = feature_cols.index(target_col)
    last12 = X[:, -12:, idx]
    if last12.shape[1] != horizon:
        return None
    return last12.copy()


# -----------------------
# Evaluate one run
# -----------------------
def evaluate_run(run_dir: Path) -> Dict:
    run_dir = Path(run_dir)

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "data_meta.json").read_text(encoding="utf-8"))

    feature_cols = meta["feature_cols"]
    target_col = meta["target_col"]
    history_len = int(cfg["history_len"])
    horizon = int(cfg["horizon"])

    # Load saved arrays
    preds_pack = np.load(run_dir / "test_predictions.npz")
    y_true = preds_pack["y_true"]
    y_pred = preds_pack["y_pred"]

    X_pack = np.load(run_dir / "x_test.npz")
    X_test = X_pack["X_test"]

    out = {
        "run_dir": str(run_dir),
        "model": cfg["model"],
        "global": metrics_dict(y_true, y_pred),
        "by_horizon": metrics_by_horizon(y_true, y_pred),
    }

    # Baselines (always available now)
    pers = baseline_persistence(X_test, feature_cols, target_col, horizon)
    out["baseline_persistence"] = metrics_dict(y_true, pers)

    seas = baseline_seasonal_naive(X_test, feature_cols, target_col, horizon, history_len)
    if seas is not None:
        out["baseline_seasonal_naive"] = metrics_dict(y_true, seas)

    # Save
    with open(run_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


# -----------------------
# Compare many runs
# -----------------------
def compare_runs(runs: List[Path]) -> pd.DataFrame:
    rows = []
    for r in runs:
        r = Path(r)
        ev_path = r / "evaluation.json"
        if ev_path.exists():
            ev = json.loads(ev_path.read_text(encoding="utf-8"))
        else:
            ev = evaluate_run(r)

        g = ev["global"]
        row = {
            "run_dir": str(r),
            "model": ev["model"],
            "MAE": g["MAE"],
            "RMSE": g["RMSE"],
            "sMAPE(%)": g["sMAPE(%)"],
            "MAPE(%)": g["MAPE(%)"],
        }

        if "baseline_persistence" in ev:
            row["PERSIST_MAE"] = ev["baseline_persistence"]["MAE"]
            row["PERSIST_RMSE"] = ev["baseline_persistence"]["RMSE"]
        if "baseline_seasonal_naive" in ev:
            row["SEASON_MAE"] = ev["baseline_seasonal_naive"]["MAE"]
            row["SEASON_RMSE"] = ev["baseline_seasonal_naive"]["RMSE"]

        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["MAE", "RMSE"], ascending=True)
    return df


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run_dir", type=str, default=None, help="results/runs/<run_name>")
    p.add_argument("--runs_root", type=str, default="results/runs")
    p.add_argument("--compare_all", action="store_true", help="evaluate + compare all runs under runs_root")
    return p.parse_args()


def main():
    args = parse_args()

    if args.compare_all:
        root = Path(args.runs_root)
        runs = [p for p in root.glob("run_*") if p.is_dir()]
        if not runs:
            raise SystemExit(f"No runs found in {root}")

        df = compare_runs(runs)

        out_dir = Path("results") / "comparisons"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "comparison_summary.csv"
        df.to_csv(out_csv, index=False)

        print(df)
        print(f"\n✅ Comparison saved: {out_csv}")
        return

    if not args.run_dir:
        raise SystemExit("Use --run_dir <path> OR --compare_all")

    res = evaluate_run(Path(args.run_dir))
    print("✅ evaluation.json saved to:", str(Path(args.run_dir) / "evaluation.json"))
    print("Global:", res["global"])


if __name__ == "__main__":
    main()
