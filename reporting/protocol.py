"""Compile the summary protocol (document 7.8)."""
import json
import pathlib
from datetime import date


def compile_protocol(results_dir: str, output_path: str,
                     all_results: dict | None = None) -> None:
    """Write the summary protocol as Markdown (doc 7.8).

    Args:
        results_dir : directory with *_summary.json files
        output_path : output file path for protocol.md
        all_results : optional in-memory results dict (avoids re-reading JSON)
    """
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Document 7.8 — Summary Protocol\n")
    lines.append(f"**Date:** {date.today().isoformat()}\n")
    lines.append("**Object:** PBM 90/250 magnetic separator  \n")
    lines.append("**Controller:** Adaptive MPC with SVR/KRR/GP kernel models  \n\n")
    lines.append("---\n\n")

    # Load summaries
    scenario_summaries = {}
    results_path = pathlib.Path(results_dir)
    if all_results:
        for sid, result in all_results.items():
            scenario_summaries[sid] = {
                "per_controller": {
                    ctrl: {"kpis": data["kpis"]}
                    for ctrl, data in result["per_controller"].items()
                }
            }
    else:
        for json_file in sorted(results_path.glob("*_summary.json")):
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            sid = data.get("scenario_id", json_file.stem)
            scenario_summaries[sid] = data

    if not scenario_summaries:
        lines.append("*No results found.*\n")
    else:
        lines.append("## KPI Summary by Scenario\n\n")
        for sid, summary in sorted(scenario_summaries.items()):
            lines.append(f"### {sid}\n\n")
            per_ctrl = summary.get("per_controller", {})
            if not per_ctrl:
                lines.append("*No controller data.*\n\n")
                continue
            lines.append("| Controller | RMSE βFe | RMSE ε | TV total | J integral |\n")
            lines.append("|---|---|---|---|---|\n")
            for ctrl_name, ctrl_data in per_ctrl.items():
                kpis = ctrl_data.get("kpis", {})
                rmse_bf = _fmt(kpis.get("rmse_beta_fe"))
                rmse_ep = _fmt(kpis.get("rmse_epsilon"))
                tv = _fmt(kpis.get("tv_total"))
                ji = _fmt(kpis.get("J_integral"))
                lines.append(f"| {ctrl_name} | {rmse_bf} | {rmse_ep} | {tv} | {ji} |\n")
            lines.append("\n")

    lines.append("---\n\n")
    lines.append("## Model Status\n\n")
    lines.append("The object model is **phenomenological**: structure is physically motivated ")
    lines.append("(Rayner & Napier-Munn 2003; GOST 10512-78); coefficients are free identification ")
    lines.append("parameters. The model is suitable for relative comparison of controllers on an ")
    lines.append("identical simulation plant; it is **not** a quantitatively validated model of ")
    lines.append("a specific production unit.\n\n")

    lines.append("## Parameters\n\n")
    lines.append("All numeric parameters are defined in `object_model/object_config.yaml` and ")
    lines.append("`config_global.yaml` with source annotations ([GOST], [LIT-QUAL], [ASSUMPTION]).\n\n")

    lines.append("## Reproducibility\n\n")
    lines.append("All RNGs initialized from `random_seed` in config. ")
    lines.append("Two identical `python run_all.py` invocations produce identical results.\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _fmt(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, dict):
        m = val.get("mean")
        if m is None:
            return "—"
        lo = val.get("ci_low")
        hi = val.get("ci_high")
        if lo is not None and hi is not None:
            return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"
        return f"{m:.4f}"
    return f"{val:.4f}" if isinstance(val, float) else str(val)
