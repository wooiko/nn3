"""Aggregation over multiple seeds: mean ± confidence interval."""
import numpy as np
from scipy import stats


def summarize(values: np.ndarray, confidence: float = 0.95) -> dict:
    """Return {'mean': float, 'ci_low': float, 'ci_high': float, 'std': float, 'n': int}."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n > 1:
        se = float(stats.sem(values))
        h = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    else:
        h = 0.0
    return {"mean": mean, "std": std, "ci_low": mean - h, "ci_high": mean + h, "n": n}


def aggregate_kpis(kpi_list: list[dict], confidence: float = 0.95) -> dict:
    """Aggregate a list of per-seed KPI dicts into mean ± CI.

    For nested dicts (e.g. mode_fractions, violations_*), aggregates each leaf.
    """
    if not kpi_list:
        return {}

    result = {}
    all_keys = kpi_list[0].keys()
    for key in all_keys:
        vals = [d[key] for d in kpi_list]
        if isinstance(vals[0], (int, float)):
            result[key] = summarize(np.array(vals, dtype=float), confidence)
        elif isinstance(vals[0], dict):
            # Recurse one level
            sub_keys = vals[0].keys()
            result[key] = {
                sk: summarize(np.array([v[sk] for v in vals], dtype=float), confidence)
                for sk in sub_keys
            }
        else:
            result[key] = vals  # non-numeric: keep list
    return result
