import os
import re
import glob
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _zfill2(x: str) -> str:
    x = str(x).strip()
    return x.zfill(2) if x.isdigit() else x


def parse_aaaamm_to_date(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    return pd.to_datetime(s, format="%Y%m", errors="coerce")


def resolve_target_column(df: pd.DataFrame, target: str = "TM") -> Tuple[pd.DataFrame, str]:
    if target in df.columns:
        return df, target

    if target.upper() == "TM":
        for alt in ["TM", "TMM"]:
            if alt in df.columns:
                return df, alt

    if "TX" in df.columns and "TN" in df.columns:
        df = df.copy()
        df["TM_DERIVED"] = (
            pd.to_numeric(df["TX"], errors="coerce") + pd.to_numeric(df["TN"], errors="coerce")
        ) / 2.0
        return df, "TM_DERIVED"

    raise ValueError(f"Target '{target}' not found and cannot be inferred from TX/TN.")


def add_month_cyc_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    m = pd.to_datetime(out[date_col]).dt.month.astype(float)
    out["month_sin"] = np.sin(2.0 * np.pi * (m / 12.0))
    out["month_cos"] = np.cos(2.0 * np.pi * (m / 12.0))
    return out


def _recursive_csv_files(data_dir: str) -> List[str]:
    patterns = [
        os.path.join(data_dir, "*.csv"),
        os.path.join(data_dir, "*.csv.gz"),
        os.path.join(data_dir, "**", "*.csv"),
        os.path.join(data_dir, "**", "*.csv.gz"),
    ]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    return sorted(set(files))


def _extract_dept_from_filename(path: str) -> Optional[str]:
    name = os.path.basename(path)
    m = re.search(r"departement[_\-\s](\d{2})", name, flags=re.IGNORECASE)
    if m:
        return _zfill2(m.group(1))
    m = re.search(r"[_\-](\d{2})[_\-]", name)
    return _zfill2(m.group(1)) if m else None


def _filter_rows_by_dept_if_possible(df: pd.DataFrame, dept_code: str) -> pd.DataFrame:
    dept_code = _zfill2(dept_code)
    candidates = ["DEPT", "DEP", "CODE_DEPT", "CODE_DEP", "DEPARTEMENT", "code_departement", "Code_departement"]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        return df
    s = df[col].astype(str).str.strip().apply(_zfill2)
    return df[s == dept_code].copy()


def load_and_preprocess_multi_dept(
    data_dir: str,
    dept_codes: List[str],
    target: str = "TM",
    features: Optional[List[str]] = None,
    station_col: str = "NUM_POSTE",
    time_col: str = "AAAAMM",
    drop_quality_cols: bool = True,
    interpolate_limit: int = 2,
    ffill_limit: int = 1,
    min_months_per_station: int = 24,
    add_month_features: bool = True,
    include_target_as_feature: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, str, List[str], str, str, str]:
    """
    Returns:
      df_clean, target_col, feature_cols_x, station_col_used, date_col, dept_col

    df_clean columns (NO duplicates):
      [DEPT_CODE, station_id, Date] + feature_cols_x
    Note: target_col is included inside feature_cols_x iff include_target_as_feature=True.
    """
    dept_codes = [_zfill2(d) for d in dept_codes]
    files = _recursive_csv_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV/CSV.GZ found under {data_dir}")

    # prefer files with departement_XX in filename if possible
    selected_files = []
    for fp in files:
        d = _extract_dept_from_filename(fp)
        if d is not None and d in set(dept_codes):
            selected_files.append(fp)
    use_files = selected_files if selected_files else files

    if verbose:
        print(f"[data_processing] data_dir={os.path.abspath(data_dir)}")
        print(f"[data_processing] requested depts={dept_codes}")
        print(f"[data_processing] found total files={len(files)} | using files={len(use_files)}")

    dfs = []
    for fp in use_files:
        try:
            df_i = pd.read_csv(fp, sep=";", compression="infer", low_memory=False)
        except Exception:
            df_i = pd.read_csv(fp, sep=",", compression="infer", low_memory=False)

        df_i.columns = [c.strip() for c in df_i.columns]

        # department tagging
        dept_from_name = _extract_dept_from_filename(fp)
        if dept_from_name is not None:
            df_i["DEPT_CODE"] = dept_from_name
        else:
            df_i["DEPT_CODE"] = np.nan

        # If dept not found in filename, try filter by dept column if exists
        if df_i["DEPT_CODE"].isna().all():
            filtered_parts = []
            for d in dept_codes:
                part = _filter_rows_by_dept_if_possible(df_i, d)
                if len(part) > 0:
                    part = part.copy()
                    part["DEPT_CODE"] = d
                    filtered_parts.append(part)
            if filtered_parts:
                df_i = pd.concat(filtered_parts, ignore_index=True)
            else:
                continue
        else:
            df_i = df_i[df_i["DEPT_CODE"].astype(str).apply(_zfill2).isin(set(dept_codes))]

        if len(df_i) > 0:
            dfs.append(df_i)

    if not dfs:
        raise FileNotFoundError("No rows loaded for the requested departments (check filenames/columns).")

    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]  # <-- critical guard

    dept_col = "DEPT_CODE"

    station_col_used = station_col
    if station_col_used not in df.columns:
        if "NOM_USUEL" in df.columns:
            station_col_used = "NOM_USUEL"
        else:
            raise ValueError(f"Station column '{station_col}' not found (and NOM_USUEL missing).")

    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found.")

    if drop_quality_cols:
        q_cols = [c for c in df.columns if str(c).upper().startswith("Q")]
        df = df.drop(columns=q_cols, errors="ignore")

    df = df.copy()
    df["Date"] = parse_aaaamm_to_date(df[time_col])
    df = df.dropna(subset=["Date", station_col_used, dept_col])

    df, target_col = resolve_target_column(df, target=target)

    # feature selection
    if not features:
        default_candidates = ["RR", "FFM", "TN", "TX", "GLOT", "INST", "LAT", "LON", "ALTI"]
        feature_cols = [c for c in default_candidates if c in df.columns and c != target_col]
        if not feature_cols:
            exclude = {station_col_used, time_col, "Date", target_col, dept_col}
            feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype != "object"]
    else:
        feature_cols = [c for c in features if c in df.columns and c != target_col]
        if not feature_cols:
            raise ValueError("Provided features not found in dataframe.")

    # numeric conversion
    for c in feature_cols + [target_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep one target column only
    base_cols = [dept_col, station_col_used, "Date"] + feature_cols + [target_col]
    base_cols = list(dict.fromkeys(base_cols))  # unique & ordered
    df = df[base_cols].copy()
    df = df.loc[:, ~df.columns.duplicated()]

    df[dept_col] = df[dept_col].astype(str).apply(_zfill2)
    df[station_col_used] = df[station_col_used].astype(str)
    df = df.sort_values([dept_col, station_col_used, "Date"])

    # duplicates by (dept, station, month)
    df = df.groupby([dept_col, station_col_used, "Date"], as_index=False).mean(numeric_only=True)

    if add_month_features:
        df = add_month_cyc_features(df, date_col="Date")

    # build X features (include target past)
    feature_cols_x = feature_cols.copy()
    if include_target_as_feature and target_col not in feature_cols_x:
        feature_cols_x.append(target_col)
    if add_month_features:
        for c in ["month_sin", "month_cos"]:
            if c not in feature_cols_x:
                feature_cols_x.append(c)
    feature_cols_x = list(dict.fromkeys(feature_cols_x))  # unique

    # model columns (only those needed for X; target is inside X if autoregressive)
    parts = []
    for (d, st), g in df.groupby([dept_col, station_col_used]):
        g = g.sort_values("Date").set_index("Date")
        if len(g) < min_months_per_station:
            continue

        idx = pd.date_range(g.index.min(), g.index.max(), freq="MS")
        g = g.reindex(idx)
        g[dept_col] = d
        g[station_col_used] = st

        # recompute month features after reindex
        g = g.reset_index().rename(columns={"index": "Date"})
        if add_month_features:
            g = add_month_cyc_features(g, date_col="Date")
        g = g.set_index("Date")

        fill_cols = feature_cols_x[:]  # includes target if include_target_as_feature
        g[fill_cols] = g[fill_cols].interpolate(method="time", limit=interpolate_limit, limit_direction="both")
        if ffill_limit and ffill_limit > 0:
            g[fill_cols] = g[fill_cols].ffill(limit=ffill_limit).bfill(limit=ffill_limit)

        g[fill_cols] = g[fill_cols].replace([np.inf, -np.inf], np.nan)
        g = g.dropna(subset=fill_cols)

        if len(g) < min_months_per_station:
            continue

        parts.append(g.reset_index())

    if not parts:
        raise ValueError("After preprocessing, no dept/station has enough valid data.")

    df_clean = pd.concat(parts, ignore_index=True)
    # IMPORTANT: do NOT append target_col again (it is already in feature_cols_x when autoregressive)
    keep_cols = [dept_col, station_col_used, "Date"] + feature_cols_x
    keep_cols = list(dict.fromkeys(keep_cols))
    df_clean = df_clean[keep_cols].copy()
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]

    if verbose:
        print(f"[data_processing] kept depts={df_clean[dept_col].nunique()} | stations={df_clean[station_col_used].nunique()}")
        print(f"[data_processing] feature_cols_x ({len(feature_cols_x)}): {feature_cols_x}")
        print(f"[data_processing] target_col={target_col}")

    return df_clean, target_col, feature_cols_x, station_col_used, "Date", dept_col


def split_by_department(
    df: pd.DataFrame,
    dept_col: str,
    train_depts: List[str],
    val_depts: List[str],
    test_depts: List[str],
) -> Dict[str, List[str]]:
    all_depts = set(df[dept_col].astype(str).apply(_zfill2).unique().tolist())
    train = [_zfill2(d) for d in train_depts if _zfill2(d) in all_depts]
    val = [_zfill2(d) for d in val_depts if _zfill2(d) in all_depts]
    test = [_zfill2(d) for d in test_depts if _zfill2(d) in all_depts]
    if not train or not test:
        raise ValueError(f"Bad dept split. Available depts={sorted(all_depts)}")
    return {"train": train, "val": val, "test": test}


def split_by_time_strict(df: pd.DataFrame, date_col: str, train_end: str, val_end: Optional[str] = None) -> Dict[str, pd.Timestamp]:
    cuts = {"train_end": pd.to_datetime(train_end)}
    if val_end is not None:
        cuts["val_end"] = pd.to_datetime(val_end)
    return cuts


def make_sequences_strict(
    df: pd.DataFrame,
    history_len: int,
    horizon: int,
    group_cols: List[str],
    time_col: str,
    feature_cols: List[str],
    target_col: str,
    y_end_max: Optional[pd.Timestamp] = None,
    y_start_min: Optional[pd.Timestamp] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str, pd.Timestamp]]]:
    """
    Strict windowing per (dept, station). Skips any window containing NaN/inf.
    Enforces leakage control via y_start_min / y_end_max.
    Returns y shape (N, H) ALWAYS.
    """
    X_list, y_list, meta = [], [], []

    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        g = g.sort_values(time_col).reset_index(drop=True)

        feats = g[feature_cols].values.astype(np.float32)

        # Defensive: if target column got duplicated in upstream, keep only first
        tgt_obj = g[target_col]
        if isinstance(tgt_obj, pd.DataFrame):
            tgt = tgt_obj.iloc[:, 0].values.astype(np.float32)
        else:
            tgt = tgt_obj.values.astype(np.float32)

        dates = pd.to_datetime(g[time_col])

        max_i = len(g) - history_len - horizon + 1
        if max_i <= 0:
            continue

        for i in range(max_i):
            y_start = dates.iloc[i + history_len]
            y_end = dates.iloc[i + history_len + horizon - 1]

            if y_start_min is not None and y_start < y_start_min:
                continue
            if y_end_max is not None and y_end > y_end_max:
                continue

            Xw = feats[i : i + history_len]
            yw = tgt[i + history_len : i + history_len + horizon]

            if not np.isfinite(Xw).all():
                continue
            if not np.isfinite(yw).all():
                continue

            dept = str(keys[0])
            station = str(keys[1]) if len(keys) > 1 else "NA"
            X_list.append(Xw)
            y_list.append(yw)
            meta.append((dept, station, y_start))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), meta


def fit_scaler_on_train(df_train: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    X = df_train[feature_cols].values
    X = np.where(np.isfinite(X), X, np.nan)
    mask = np.isfinite(X).all(axis=1)
    scaler.fit(X[mask])
    return scaler


def apply_scaler(df: pd.DataFrame, feature_cols: List[str], scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    X = out[feature_cols].values
    X = scaler.transform(X)
    X = np.where(np.isfinite(X), X, np.nan)
    out[feature_cols] = X
    return out



class WeatherDataProcessor:
    #Compat layer for the dashboard.
    

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir


    def get_available_departments(self):
        files = _recursive_csv_files(self.data_dir)
        dept_codes = set()
        for fp in files:
            d = _extract_dept_from_filename(fp)
            if d is not None:
                dept_codes.add(d)
        return sorted(dept_codes)


    def load_dept_data(self, dept_code: str):
        # Use your new pipeline (single dept)
        df, target_col, feature_cols_x, station_col, time_col, dept_col = load_and_preprocess_multi_dept(
            data_dir=self.data_dir,
            dept_codes=[str(dept_code)],
            target="TM",
            # Keep default features OR specify if you want
            verbose=False,
        )

        # Ensure we have a usable Date column for the dashboard
        # AAAAMM is like 202401 -> convert to first day of month
        if "AAAAMM" in df.columns:
            s = df["AAAAMM"].astype(str)
            df["Date"] = pd.to_datetime(s + "01", format="%Y%m%d", errors="coerce")
        elif time_col in df.columns:
            df["Date"] = pd.to_datetime(df[time_col], errors="coerce")
        else:
            raise ValueError("No time column found to build Date for dashboard.")

        # Rename columns to match visualization.py expectations
        rename_map = {
            "TM": "Temperature",
            "TN": "Temp_Min",
            "TX": "Temp_Max",
            "RR": "Precipitation",
            "FFM": "Wind",
            "INST": "Ensoleillement",
        }
        for k, v in rename_map.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]

        # Keep only useful columns + Date
        keep = ["Date", "Temperature", "Temp_Min", "Temp_Max", "Ensoleillement", "Precipitation", "Wind"]
        keep = [c for c in keep if c in df.columns]
        out = df[keep].dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        return out


