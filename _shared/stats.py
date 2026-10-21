"""Metrics + bootstrap CIs (seeded)."""
import numpy as np

def ece(conf, correct, bins=15):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)

def brier(P, y_idx):
    P = np.asarray(P, float); Y = np.zeros_like(P); Y[np.arange(len(y_idx)), y_idx] = 1
    return float(((P - Y) ** 2).sum(1).mean())

def macro_f1(y, p, labels):
    y = np.asarray(y); p = np.asarray(p); f = []
    for l in labels:
        tp = ((p == l) & (y == l)).sum(); fp = ((p == l) & (y != l)).sum(); fn = ((p != l) & (y == l)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))

def bootstrap(fn, n, B=2000, seed=0, strata=None):
    """fn(idx)->float. Returns (point, lo, hi) percentile 95% CI. Optional stratified resampling."""
    rng = np.random.default_rng(seed); idx_all = np.arange(n)
    point = fn(idx_all); vals = []
    if strata is not None:
        strata = np.asarray(strata); groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    for _ in range(B):
        if strata is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        vals.append(fn(idx))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)

def fmt_ci(t, pct=False, d=3):
    p, lo, hi = t
    if pct:
        return f"{100*p:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"
    return f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"
