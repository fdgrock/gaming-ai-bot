"""UI helper utilities used by Streamlit app and unit tests.

Functions here are pure-Python (no Streamlit imports) so tests can import them
without launching the UI.
"""
from typing import Any, Dict, List, Optional
import glob
import os
import json
import pandas as pd


def number_frequency(game_name: str) -> pd.DataFrame:
    """Return a dataframe with columns ['number','count'] for historic draws.

    Searches under data/<game_name>/*.csv and data/<game_name>/history/*.csv and
    falls back to any CSVs under data/ if needed.
    """
    dfs = []
    candidates = glob.glob(os.path.join('data', game_name, '*.csv'))
    candidates.extend(glob.glob(os.path.join('data', game_name, 'history', '*.csv')))
    candidates.extend(glob.glob(os.path.join('data', '**', '*.csv'), recursive=True))
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        try:
            df = pd.read_csv(p)
            if 'numbers' in df.columns:
                dfs.append(df[['draw_date', 'numbers']])
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    s = df['numbers'].dropna().astype(str).str.split(',')
    s_exploded = s.explode().str.strip().astype(int)
    freq = s_exploded.value_counts().reset_index()
    freq.columns = ['number', 'count']
    return freq.sort_values('number')


def find_attention_in_predictions(game_name: str, limit: int = 5) -> Optional[Dict[str, Any]]:
    """Search recent predictions cache for an attention matrix.

    Returns a dict with keys: 'file', 'attention' (2D list), and 'top_tokens' (list of token indices sorted by aggregate attention).
    If none found, returns None.
    """
    pred_dir = os.path.join('predictions', game_name)
    if not os.path.isdir(pred_dir):
        return None
    files = sorted(glob.glob(os.path.join(pred_dir, '*.json')), reverse=True)
    from ai_lottery_bot.model_manager.manager import extract_attention_from_model

    for p in files[:limit]:
        try:
            with open(p, 'r') as f:
                obj = json.load(f)
            att = None
            if isinstance(obj, dict):
                if 'attention' in obj:
                    att = obj['attention']
                elif 'attention_map' in obj:
                    att = obj['attention_map']
                # If the prediction file references a saved model name, try to load
                # the model and use the model_manager helper to extract attention.
                elif 'model_name' in obj:
                    try:
                        mname = obj.get('model_name')
                        # best-effort load model from models/ (joblib)
                        from ai_lottery_bot.model_manager import manager as _mm
                        try:
                            mm = _mm.load_model(mname)
                            att = extract_attention_from_model(mm)
                        except Exception:
                            att = None
                    except Exception:
                        att = None
            if att is None and isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and 'attention' in item:
                        att = item['attention']
                        break
            if att is None:
                continue
            # normalize and compute top tokens
            try:
                import numpy as _np
                arr = _np.array(att)
                col_sum = arr.sum(axis=0)
                top_idx = list(_np.argsort(col_sum)[::-1])
            except Exception:
                top_idx = []
            return {'file': p, 'attention': att, 'top_tokens': top_idx}
        except Exception:
            continue
    return None


def load_recent_metrics(game_name: str, n: int = 100) -> pd.DataFrame:
    """Load recent metrics from metrics/<game>.csv if present."""
    metrics_file = os.path.join('metrics', f"{game_name}.csv")
    if not os.path.exists(metrics_file):
        return pd.DataFrame()
    try:
        df = pd.read_csv(metrics_file)
        return df.tail(n).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()
