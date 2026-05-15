"""Integration tests for object model (spec 2.2.6, static_model_spec.md sec. 8)."""
import warnings
import numpy as np
import pytest
import yaml
from pathlib import Path

from object_model.static_model import predict_outputs
from object_model.dynamic_model import DynamicModel
from object_model.validation_protocol import run as run_validation

CFG_PATH = Path(__file__).parents[2] / "object_model" / "object_config.yaml"


@pytest.fixture(scope="module")
def cfg():
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Validation protocol ───────────────────────────────────────────────────────

def test_validation_protocol_all_checks_pass(cfg):
    result = run_validation(cfg)
    failed = [cid for cid, r in result["results"].items() if not r["passed"]]
    assert result["passed"], f"Validation checks failed: {failed}\n" + "\n".join(
        f"  {cid} ({result['results'][cid]['name']}): {result['results'][cid]['detail']}"
        for cid in failed
    )


def test_output_ranges(cfg):
    """8.1: outputs in [0,1] on domain corners."""
    corners = [
        (cfg["B_min_dom"], cfg["Q_min_dom"], cfg["rho_min_dom"], cfg["alpha_fe_min_dom"]),
        (cfg["B_max_dom"], cfg["Q_max_dom"], cfg["rho_max_dom"], cfg["alpha_fe_max_dom"]),
        (cfg["B_min_dom"], cfg["Q_max_dom"], cfg["rho_max_dom"], cfg["alpha_fe_min_dom"]),
        (cfg["B_max_dom"], cfg["Q_min_dom"], cfg["rho_min_dom"], cfg["alpha_fe_max_dom"]),
    ]
    for B, Q, rho, af in corners:
        bf, ep = predict_outputs(B, rho, Q, af, cfg)
        assert 0.0 <= bf <= 1.0, f"βFe={bf} out of range at B={B},Q={Q},ρ={rho},αFe={af}"
        assert 0.0 <= ep <= 1.0, f"ε={ep} out of range at B={B},Q={Q},ρ={rho},αFe={af}"


def test_monotonicity_epsilon_vs_B(cfg):
    """8.2: ε non-decreasing in B."""
    nom = {k: (cfg[k + "_min_dom"] + cfg[k + "_max_dom"]) / 2 for k in ("Q", "rho")}
    Bs = np.linspace(cfg["B_min_dom"], cfg["B_max_dom"], 30)
    eps = [predict_outputs(B, nom["rho"], nom["Q"], cfg["alpha_fe_ref"], cfg)[1] for B in Bs]
    diffs = np.diff(eps)
    assert np.all(diffs >= -1e-9), f"ε(B) not non-decreasing; min diff={diffs.min():.6f}"


def test_monotonicity_epsilon_vs_Q(cfg):
    """8.3: ε non-increasing in Q."""
    nom = {k: (cfg[k + "_min_dom"] + cfg[k + "_max_dom"]) / 2 for k in ("B", "rho")}
    Qs = np.linspace(cfg["Q_min_dom"], cfg["Q_max_dom"], 30)
    eps = [predict_outputs(nom["B"], nom["rho"], Q, cfg["alpha_fe_ref"], cfg)[1] for Q in Qs]
    diffs = np.diff(eps)
    assert np.all(diffs <= 1e-9), f"ε(Q) not non-increasing; max diff={diffs.max():.6f}"


def test_monotonicity_beta_fe_vs_rho(cfg):
    """8.5: βFe non-increasing in ρ."""
    nom = {k: (cfg[k + "_min_dom"] + cfg[k + "_max_dom"]) / 2 for k in ("B", "Q")}
    rhos = np.linspace(cfg["rho_min_dom"], cfg["rho_max_dom"], 30)
    bfs = [predict_outputs(nom["B"], rho, nom["Q"], cfg["alpha_fe_ref"], cfg)[0] for rho in rhos]
    diffs = np.diff(bfs)
    assert np.all(diffs <= 1e-9), f"βFe(ρ) not non-increasing; max diff={diffs.max():.6f}"


def test_antagonism_exists(cfg):
    """8.7: βFe and ε have opposite ∂/∂B somewhere in domain."""
    nom = {k: (cfg[k + "_min_dom"] + cfg[k + "_max_dom"]) / 2 for k in ("Q", "rho")}
    Bs = np.linspace(cfg["B_min_dom"], cfg["B_max_dom"], 40)
    bfs = np.array([predict_outputs(B, nom["rho"], nom["Q"], cfg["alpha_fe_ref"], cfg)[0] for B in Bs])
    eps = np.array([predict_outputs(B, nom["rho"], nom["Q"], cfg["alpha_fe_ref"], cfg)[1] for B in Bs])
    d_bf = np.diff(bfs)
    d_ep = np.diff(eps)
    mask = (np.abs(d_bf) > 1e-9) & (np.abs(d_ep) > 1e-9)
    assert mask.any(), "Both derivatives near zero everywhere — cannot detect antagonism"
    antagonism = np.any(np.sign(d_bf[mask]) != np.sign(d_ep[mask]))
    assert antagonism, "No antagonism βFe↔ε detected in B sweep"


def test_boundary_modes_clip_warning(cfg):
    """Spec 7.2: out-of-domain input raises UserWarning and clips, not crash."""
    B_out = cfg["B_max_dom"] + 0.05
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bf, ep = predict_outputs(B_out, cfg["rho_min_dom"], cfg["Q_min_dom"], cfg["alpha_fe_ref"], cfg)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
    assert 0.0 <= bf <= 1.0
    assert 0.0 <= ep <= 1.0


def test_nominal_regime_plausible(cfg):
    """8.10: nominal operating point gives plausible output values."""
    B_nom = (cfg["B_min_dom"] + cfg["B_max_dom"]) / 2
    Q_nom = (cfg["Q_min_dom"] + cfg["Q_max_dom"]) / 2
    rho_nom = (cfg["rho_min_dom"] + cfg["rho_max_dom"]) / 2
    bf, ep = predict_outputs(B_nom, rho_nom, Q_nom, cfg["alpha_fe_ref"], cfg)
    assert 0.58 <= bf <= 0.70, f"βFe={bf:.4f} outside plausible range [0.58, 0.70]"
    assert 0.80 <= ep <= 1.00, f"ε={ep:.4f} outside plausible range [0.80, 1.00]"


# ── Dynamic model ─────────────────────────────────────────────────────────────

def test_dynamic_model_reaches_steady_state(cfg):
    """Dynamic model output converges to static model at steady input."""
    model = DynamicModel(cfg)
    B = (cfg["B_min_dom"] + cfg["B_max_dom"]) / 2
    Q = (cfg["Q_min_dom"] + cfg["Q_max_dom"]) / 2
    rho = (cfg["rho_min_dom"] + cfg["rho_max_dom"]) / 2
    af = cfg["alpha_fe_ref"]
    model.reset(B, rho, Q, af)

    bf_static, ep_static = predict_outputs(B, rho, Q, af, cfg)
    for _ in range(50):
        y = model.step(B, rho, Q, af)

    assert abs(y[0] - bf_static) < 1e-6, f"βFe did not converge: {y[0]:.6f} vs {bf_static:.6f}"
    assert abs(y[1] - ep_static) < 1e-6, f"ε did not converge: {y[1]:.6f} vs {ep_static:.6f}"


def test_dynamic_model_delay(cfg):
    """Step change at t=0 should not appear in output until after delay steps."""
    model = DynamicModel(cfg)
    B_lo = cfg["B_min_dom"]
    B_hi = cfg["B_max_dom"]
    Q = (cfg["Q_min_dom"] + cfg["Q_max_dom"]) / 2
    rho = (cfg["rho_min_dom"] + cfg["rho_max_dom"]) / 2
    af = cfg["alpha_fe_ref"]

    model.reset(B_lo, rho, Q, af)
    y_before = model.step(B_lo, rho, Q, af)

    # Step to high B — output of ε should not change before delay steps
    delay_ep = cfg["delay_epsilon"]
    outputs = []
    for _ in range(delay_ep):
        outputs.append(model.step(B_hi, rho, Q, af))

    # Within delay, ε output should equal (or be close to) pre-step steady state
    eps_before = y_before[1]
    for i, y in enumerate(outputs):
        assert abs(y[1] - eps_before) < 1e-4, (
            f"ε changed before delay at step {i+1}: {y[1]:.6f} vs {eps_before:.6f}"
        )
