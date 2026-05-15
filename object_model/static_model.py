"""Static phenomenological model: βFe = f(B,Q,ρ,αFe), ε = f(B,Q,ρ,αFe).

Structure justified by magnetic-force / Stokes-drag balance (pbm_report_1, theoretical section).
Coefficients are free identification parameters — see object_config.yaml for status and sources.
Reference: Rayner & Napier-Munn (2003) for ε base structure; static_model_spec.md for full rationale.
"""
import warnings
import numpy as np


def _sat_B(B: float, B_min_sat: float, B_scale: float) -> float:
    """Saturation function for magnetic induction effect on capture rate."""
    if B <= B_min_sat:
        return 0.0
    return 1.0 - np.exp(-(B - B_min_sat) / B_scale)


def _g_rho(rho: float, rho_opt: float, rho_width: float) -> float:
    """Bell-shaped pulp density modifier for ε (non-monotone, optimum at rho_opt)."""
    return np.exp(-0.5 * ((rho - rho_opt) / rho_width) ** 2)


def _g_alpha(alpha_fe: float, alpha_fe_ref: float, c_alpha: float) -> float:
    """Magnetic capacity overload factor — degrades ε when αFe exceeds reference."""
    return 1.0 - c_alpha * max(0.0, alpha_fe - alpha_fe_ref)


def _h_B(B: float, B_opt_beta: float, B_drop_beta: float) -> float:
    """Asymmetric bell for βFe(B): rises to B_opt_beta then slowly falls (inclusion of middlings)."""
    rise = _sat_B(B, 0.0, B_opt_beta)          # saturating rise toward B_opt_beta
    drop = np.exp(-B_drop_beta * max(0.0, B - B_opt_beta))
    return rise * drop


def _p_Q(Q: float, Q_opt_beta: float, Q_width_beta: float) -> float:
    """Shallow bell for βFe(Q): weak wash-selectivity maximum, nearly monotone in GOST range."""
    return np.exp(-0.5 * ((Q - Q_opt_beta) / Q_width_beta) ** 2)


def _p_rho(rho: float, rho_min_dom: float, rho_drop_beta: float) -> float:
    """Monotonically decreasing density penalty for βFe (viscosity, middling entrapment)."""
    return np.exp(-rho_drop_beta * (rho - rho_min_dom))


def _check_domain(B, Q, rho, alpha_fe, cfg) -> bool:
    in_domain = (
        cfg["B_min_dom"] <= B <= cfg["B_max_dom"]
        and cfg["Q_min_dom"] <= Q <= cfg["Q_max_dom"]
        and cfg["rho_min_dom"] <= rho <= cfg["rho_max_dom"]
        and cfg["alpha_fe_min_dom"] <= alpha_fe <= cfg["alpha_fe_max_dom"]
    )
    return in_domain


def predict_outputs(
    B: float,
    rho: float,
    Q: float,
    alpha_fe: float,
    cfg: dict,
) -> tuple[float, float]:
    """Compute static (βFe, ε) for given operating point.

    Returns:
        (beta_fe, epsilon) — fractions in [0, 1].

    Out-of-domain inputs are clipped to domain boundaries and a warning is raised.
    The caller (mpc_controller) should handle the warning as appropriate.
    """
    if not _check_domain(B, Q, rho, alpha_fe, cfg):
        warnings.warn(
            f"Input out of domain: B={B}, Q={Q}, rho={rho}, alpha_fe={alpha_fe}. "
            "Clipping to domain boundaries.",
            stacklevel=2,
        )
        B = np.clip(B, cfg["B_min_dom"], cfg["B_max_dom"])
        Q = np.clip(Q, cfg["Q_min_dom"], cfg["Q_max_dom"])
        rho = np.clip(rho, cfg["rho_min_dom"], cfg["rho_max_dom"])
        alpha_fe = np.clip(alpha_fe, cfg["alpha_fe_min_dom"], cfg["alpha_fe_max_dom"])

    # ── ε model ─────────────────────────────────────────────────────────────
    tau = cfg["tau_ref"] * (cfg["Q_ref"] / Q)
    k_eff = cfg["k_0"] * _sat_B(B, cfg["B_min_sat"], cfg["B_scale"])
    g_rho_val = _g_rho(rho, cfg["rho_opt"], cfg["rho_width"])
    g_alpha_val = _g_alpha(alpha_fe, cfg["alpha_fe_ref"], cfg["c_alpha"])

    epsilon = cfg["epsilon_max"] * (1.0 - np.exp(-k_eff * tau)) * g_rho_val * g_alpha_val

    # ── βFe model ────────────────────────────────────────────────────────────
    h_B_val = _h_B(B, cfg["B_opt_beta"], cfg["B_drop_beta"])
    p_Q_val = _p_Q(Q, cfg["Q_opt_beta"], cfg["Q_width_beta"])
    p_rho_val = _p_rho(rho, cfg["rho_min_dom"], cfg["rho_drop_beta"])

    beta_fe = (
        cfg["beta_fe_floor"]
        + (cfg["beta_fe_ceil"] - cfg["beta_fe_floor"]) * h_B_val * p_Q_val * p_rho_val
        + cfg["c_beta"] * (alpha_fe - cfg["alpha_fe_ref"])
    )

    # ── Physical bounds clip (spec 7.3) ──────────────────────────────────────
    beta_fe = float(np.clip(beta_fe, 0.0, 1.0))
    epsilon = float(np.clip(epsilon, 0.0, 1.0))

    return beta_fe, epsilon
