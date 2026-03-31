import numpy as np
import pandas as pd
from src.data_processing import make_sequences_strict

def test_make_sequences_strict_shapes():
    n_steps = 60
    history_len = 24
    horizon = 12

    df = pd.DataFrame({
        "feature1": np.random.randn(n_steps),
        "feature2": np.random.randn(n_steps),
        "target": np.random.randn(n_steps),
    })

    X, y = make_sequences_strict(
        df=df,
        feature_cols=["feature1", "feature2"],
        target_col="target",
        history_len=history_len,
        horizon=horizon
    )

    assert X.ndim == 3
    assert y.ndim == 2
    assert X.shape[1] == history_len
    assert X.shape[2] == 2
    assert y.shape[1] == horizon