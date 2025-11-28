import pandas as pd
from typing import Any


def extract_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder feature extraction: counts and rolling stats."""
    out = df.copy()
    # example: create a column with sum of numeric columns
    nums = out.select_dtypes(include="number")
    if not nums.empty:
        out["_num_sum"] = nums.sum(axis=1)
    return out
