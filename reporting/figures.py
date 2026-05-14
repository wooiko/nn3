"""Generate figures: transient responses, σ²_GP trajectory, R(k) curve (docs 7.2–7.4)."""
import pathlib
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless runs
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def _save_or_skip(fig, path: pathlib.Path) -> None:
    if _MPL_AVAILABLE:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def transient_response(results: dict, scenario_id: str, output_dir: str) -> None:
    """Plot βFe and ε trajectories for all controllers on the first seed (doc 7.2)."""
    if not _MPL_AVAILABLE or scenario_id not in results:
        return
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = results[scenario_id]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].set_ylabel("βFe [frac]")
    axes[1].set_ylabel("ε [frac]")
    axes[1].set_xlabel("Step k")
    fig.suptitle(f"{scenario_id} — Transient Response")

    colors = {"full": "tab:blue", "classic_mpc": "tab:orange", "prototype_mpc": "tab:green"}
    y_ref = None

    for ctrl_name, ctrl_data in result["per_controller"].items():
        seeds = ctrl_data.get("seeds", [])
        if not seeds:
            continue
        trace = seeds[0]
        y = np.array(trace["y"])
        y_ref = trace["y_ref"]
        color = colors.get(ctrl_name, "gray")
        axes[0].plot(y[:, 0], label=ctrl_name, color=color, alpha=0.8)
        axes[1].plot(y[:, 1], label=ctrl_name, color=color, alpha=0.8)

    if y_ref is not None:
        axes[0].axhline(y_ref[0], color="red", linestyle="--", label="ref")
        axes[1].axhline(y_ref[1], color="red", linestyle="--", label="ref")

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    _save_or_skip(fig, out / f"{scenario_id}_transient.png")


def gp_variance_trajectory(results: dict, scenario_id: str, output_dir: str) -> None:
    """Plot σ²_GP trajectory and mode switches for the full controller (doc 7.3)."""
    if not _MPL_AVAILABLE or scenario_id not in results:
        return
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = results[scenario_id]
    ctrl_data = result["per_controller"].get("full")
    if not ctrl_data or not ctrl_data.get("seeds"):
        return

    trace = ctrl_data["seeds"][0]
    sigma2 = np.array(trace.get("sigma2", []))
    modes = trace.get("modes", [])
    if not len(sigma2):
        return

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(sigma2, color="tab:blue", label="σ²_GP")

    # Shade regions by mode
    mode_colors = {"nominal": "white", "cautious": "lightyellow", "conservative": "mistyrose"}
    prev_mode = modes[0] if modes else "nominal"
    seg_start = 0
    for k, m in enumerate(modes):
        if m != prev_mode or k == len(modes) - 1:
            ax.axvspan(seg_start, k, alpha=0.3,
                       color=mode_colors.get(prev_mode, "white"), label=None)
            seg_start = k
            prev_mode = m

    ax.set_xlabel("Step k")
    ax.set_ylabel("σ²_GP")
    ax.set_title(f"{scenario_id} — GP Variance Trajectory")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_or_skip(fig, out / f"{scenario_id}_gp_variance.png")


def r_vs_sigma2(results: dict, scenario_id: str, output_dir: str) -> None:
    """Plot R diagonal vs σ²_GP (doc 7.4) — parametric curve from first seed trace."""
    if not _MPL_AVAILABLE or scenario_id not in results:
        return
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = results[scenario_id]
    cfg = result.get("cfg", {})
    ctrl_data = result["per_controller"].get("full")
    if not ctrl_data or not ctrl_data.get("seeds"):
        return

    trace = ctrl_data["seeds"][0]
    sigma2 = np.array(trace.get("sigma2", []))
    modes = trace.get("modes", [])
    if not len(sigma2):
        return

    # Reconstruct R_diag from logged modes and sigma2
    R0 = float(cfg.get("R0") or 1.0)
    alpha_r = float(cfg.get("alpha_r") or 1.0)
    gamma_r = float(cfg.get("gamma_r") or 5.0)

    R_diag = np.where(
        np.array(modes) == "nominal", R0,
        np.where(np.array(modes) == "cautious",
                 R0 + alpha_r * sigma2,
                 gamma_r * R0)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(sigma2, R_diag, c=np.arange(len(sigma2)), cmap="viridis",
                    s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Step k")
    ax.set_xlabel("σ²_GP")
    ax.set_ylabel("R diagonal")
    ax.set_title(f"{scenario_id} — R(k) vs σ²_GP")
    plt.tight_layout()
    _save_or_skip(fig, out / f"{scenario_id}_r_vs_sigma2.png")
