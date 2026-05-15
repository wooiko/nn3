"""Validation protocol for the static object model (spec 2.2.6, static_model_spec.md sec. 8).

Checks 8.1–8.10:
  8.1  Output ranges on full domain grid
  8.2  ε(B) monotonically non-decreasing
  8.3  ε(Q) monotonically non-increasing
  8.4  βFe(B) has a maximum (or near-monotone with shallow character)
  8.5  βFe(ρ) monotonically non-increasing
  8.6  ε(ρ) has optimum or is non-increasing (consistent with g_rho)
  8.7  Antagonism βFe↔ε exists (∂βFe/∂x and ∂ε/∂x have opposite signs somewhere)
  8.8  Interaction of arguments (∂²ε/∂B∂ρ ≠ 0  ⟹ not purely additive)
  8.9  Continuity — no jumps on the domain grid
  8.10 Nominal regime gives physically plausible βFe, ε

Returns a dict with keys: 'passed' (bool), 'results' (dict check_id → dict).
"""
import numpy as np
from object_model.static_model import predict_outputs


# ── Grid resolution ───────────────────────────────────────────────────────────
_N = 20  # points per axis for 1-D sweeps
_N_CROSS = 8  # points per axis for 2-D cross-derivative check


def _nominal(cfg: dict) -> dict:
    return {
        "B": (cfg["B_min_dom"] + cfg["B_max_dom"]) / 2,
        "Q": (cfg["Q_min_dom"] + cfg["Q_max_dom"]) / 2,
        "rho": (cfg["rho_min_dom"] + cfg["rho_max_dom"]) / 2,
        "alpha_fe": cfg["alpha_fe_ref"],
    }


def _sweep(vary: str, cfg: dict, n: int = _N):
    """Return arrays (x_vals, beta_fe_vals, epsilon_vals) sweeping one variable."""
    nom = _nominal(cfg)
    ranges = {
        "B": np.linspace(cfg["B_min_dom"], cfg["B_max_dom"], n),
        "Q": np.linspace(cfg["Q_min_dom"], cfg["Q_max_dom"], n),
        "rho": np.linspace(cfg["rho_min_dom"], cfg["rho_max_dom"], n),
        "alpha_fe": np.linspace(cfg["alpha_fe_min_dom"], cfg["alpha_fe_max_dom"], n),
    }
    xs = ranges[vary]
    beta_vals, eps_vals = [], []
    for x in xs:
        kw = dict(nom)
        kw[vary] = float(x)
        bf, ep = predict_outputs(kw["B"], kw["rho"], kw["Q"], kw["alpha_fe"], cfg)
        beta_vals.append(bf)
        eps_vals.append(ep)
    return xs, np.array(beta_vals), np.array(eps_vals)


def _check_81(cfg: dict) -> dict:
    """8.1 Output ranges: both outputs in [0,1] on random domain sample."""
    rng = np.random.default_rng(0)
    n_pts = 500
    Bs = rng.uniform(cfg["B_min_dom"], cfg["B_max_dom"], n_pts)
    Qs = rng.uniform(cfg["Q_min_dom"], cfg["Q_max_dom"], n_pts)
    rhos = rng.uniform(cfg["rho_min_dom"], cfg["rho_max_dom"], n_pts)
    afs = rng.uniform(cfg["alpha_fe_min_dom"], cfg["alpha_fe_max_dom"], n_pts)
    bfs, eps = [], []
    for B, Q, rho, af in zip(Bs, Qs, rhos, afs):
        bf, ep = predict_outputs(B, rho, Q, af, cfg)
        bfs.append(bf)
        eps.append(ep)
    bfs = np.array(bfs)
    eps = np.array(eps)
    ok = bool(np.all(bfs >= 0) and np.all(bfs <= 1) and np.all(eps >= 0) and np.all(eps <= 1))
    return {"passed": ok, "detail": f"beta_fe: [{bfs.min():.4f}, {bfs.max():.4f}], eps: [{eps.min():.4f}, {eps.max():.4f}]"}


def _check_82(cfg: dict) -> dict:
    """8.2 ε(B) monotonically non-decreasing."""
    _, _, eps = _sweep("B", cfg)
    diffs = np.diff(eps)
    ok = bool(np.all(diffs >= -1e-9))
    return {"passed": ok, "detail": f"min diff={diffs.min():.6f}"}


def _check_83(cfg: dict) -> dict:
    """8.3 ε(Q) monotonically non-increasing."""
    _, _, eps = _sweep("Q", cfg)
    diffs = np.diff(eps)
    ok = bool(np.all(diffs <= 1e-9))
    return {"passed": ok, "detail": f"max diff={diffs.max():.6f}"}


def _check_84(cfg: dict) -> dict:
    """8.4 βFe(B) has a maximum inside or near the domain (∂βFe/∂B changes sign)."""
    _, bfs, _ = _sweep("B", cfg)
    diffs = np.diff(bfs)
    has_max = bool(np.any(diffs > 0) and np.any(diffs < 0))
    has_mono_rise = bool(np.all(diffs >= -1e-9))  # acceptable if purely rising (spec allows)
    ok = has_max or has_mono_rise
    return {"passed": ok, "detail": f"has_maximum={has_max}, monotone_rise={has_mono_rise}"}


def _check_85(cfg: dict) -> dict:
    """8.5 βFe(ρ) monotonically non-increasing."""
    _, bfs, _ = _sweep("rho", cfg)
    diffs = np.diff(bfs)
    ok = bool(np.all(diffs <= 1e-9))
    return {"passed": ok, "detail": f"max diff={diffs.max():.6f}"}


def _check_86(cfg: dict) -> dict:
    """8.6 ε(ρ) consistent with g_rho (has optimum or non-increasing)."""
    _, _, eps = _sweep("rho", cfg)
    idx_max = int(np.argmax(eps))
    # Optimum inside domain, or monotone (both acceptable per spec 6.2)
    has_optimum = 0 < idx_max < len(eps) - 1
    is_nonincreasing = bool(np.all(np.diff(eps) <= 1e-9))
    ok = has_optimum or is_nonincreasing
    return {"passed": ok, "detail": f"idx_max={idx_max}/{len(eps)-1}, has_optimum={has_optimum}, nonincreasing={is_nonincreasing}"}


def _check_87(cfg: dict) -> dict:
    """8.7 Antagonism βFe↔ε: exists argument where sign(∂βFe/∂x) ≠ sign(∂ε/∂x)."""
    antagonisms = []
    for var in ("B", "Q", "rho", "alpha_fe"):
        _, bfs, eps = _sweep(var, cfg)
        d_bf = np.diff(bfs)
        d_ep = np.diff(eps)
        # Mask out near-zero (flat) derivatives
        mask = (np.abs(d_bf) > 1e-9) & (np.abs(d_ep) > 1e-9)
        if mask.any() and np.any(np.sign(d_bf[mask]) != np.sign(d_ep[mask])):
            antagonisms.append(var)
    ok = len(antagonisms) > 0
    return {"passed": ok, "detail": f"antagonism found in: {antagonisms}"}


def _check_88(cfg: dict) -> dict:
    """8.8 Interaction: ∂²ε/∂B∂ρ ≠ 0 (not purely additive)."""
    n = _N_CROSS
    Bs = np.linspace(cfg["B_min_dom"], cfg["B_max_dom"], n)
    rhos = np.linspace(cfg["rho_min_dom"], cfg["rho_max_dom"], n)
    nom = _nominal(cfg)
    eps_grid = np.zeros((n, n))
    for i, B in enumerate(Bs):
        for j, rho in enumerate(rhos):
            _, ep = predict_outputs(B, rho, nom["Q"], nom["alpha_fe"], cfg)
            eps_grid[i, j] = ep
    d2_BRho = np.diff(np.diff(eps_grid, axis=0), axis=1)
    max_cross = float(np.abs(d2_BRho).max())
    ok = max_cross > 1e-9
    return {"passed": ok, "detail": f"max |∂²ε/∂B∂ρ| = {max_cross:.2e}"}


def _check_89(cfg: dict) -> dict:
    """8.9 Continuity: no jumps larger than a physical threshold on domain grid."""
    jump_threshold = 0.05  # 5 pp jump would be unphysical
    for var in ("B", "Q", "rho"):
        xs, bfs, eps = _sweep(var, cfg, n=200)
        if np.any(np.abs(np.diff(bfs)) > jump_threshold) or np.any(np.abs(np.diff(eps)) > jump_threshold):
            return {"passed": False, "detail": f"Discontinuity detected sweeping {var}"}
    return {"passed": True, "detail": "No jumps above threshold on B, Q, rho sweeps"}


def _check_810(cfg: dict) -> dict:
    """8.10 Nominal regime gives physically plausible βFe ∈ [0.58, 0.70], ε ∈ [0.80, 1.0]."""
    nom = _nominal(cfg)
    bf, ep = predict_outputs(nom["B"], nom["rho"], nom["Q"], nom["alpha_fe"], cfg)
    bf_ok = 0.58 <= bf <= 0.70
    ep_ok = 0.80 <= ep <= 1.00
    ok = bf_ok and ep_ok
    return {"passed": ok, "detail": f"nominal βFe={bf:.4f}, ε={ep:.4f}; expected βFe∈[0.58,0.70], ε∈[0.80,1.00]"}


_CHECKS = {
    "8.1": ("Output ranges on domain sample", _check_81),
    "8.2": ("ε(B) non-decreasing", _check_82),
    "8.3": ("ε(Q) non-increasing", _check_83),
    "8.4": ("βFe(B) has maximum or rises", _check_84),
    "8.5": ("βFe(ρ) non-increasing", _check_85),
    "8.6": ("ε(ρ) has optimum or non-increasing", _check_86),
    "8.7": ("Antagonism βFe↔ε exists", _check_87),
    "8.8": ("Argument interaction (not purely additive)", _check_88),
    "8.9": ("Continuity (no jumps)", _check_89),
    "8.10": ("Nominal regime plausible", _check_810),
}


def run(cfg: dict) -> dict:
    """Run all validation checks 8.1–8.10.

    Returns:
        {
          'passed': bool,  # True only if ALL checks passed
          'results': {
              '8.1': {'name': ..., 'passed': bool, 'detail': str},
              ...
          }
        }
    """
    results = {}
    all_passed = True
    for check_id, (name, fn) in _CHECKS.items():
        res = fn(cfg)
        res["name"] = name
        results[check_id] = res
        if not res["passed"]:
            all_passed = False
    return {"passed": all_passed, "results": results}
