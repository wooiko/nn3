"""Generate KPI tables (CSV + formatted) for all scenarios (docs 7.1, 7.5, 7.6)."""
import csv
import pathlib


def kpi_table(results: dict, output_dir: str) -> None:
    """Write KPI comparison table to CSV and plain-text Markdown.

    Args:
        results    : {scenario_id: run_scenario_result_dict}
        output_dir : directory to write kpi_table.csv and kpi_table.md
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario_id, result in results.items():
        for ctrl_name, ctrl_data in result["per_controller"].items():
            kpis = ctrl_data["kpis"]
            row = {
                "scenario": scenario_id,
                "controller": ctrl_name,
                "rmse_beta_fe": _fmt(kpis.get("rmse_beta_fe")),
                "rmse_epsilon": _fmt(kpis.get("rmse_epsilon")),
                "tv_total": _fmt(kpis.get("tv_total")),
                "J_integral": _fmt(kpis.get("J_integral")),
                "n_filtered": _fmt(kpis.get("n_filtered")),
                "p_nominal": _fmt(_nested(kpis, "mode_fractions", "nominal")),
                "p_cautious": _fmt(_nested(kpis, "mode_fractions", "cautious")),
                "p_conservative": _fmt(_nested(kpis, "mode_fractions", "conservative")),
            }
            rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    # CSV
    csv_path = out / "kpi_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Markdown
    md_path = out / "kpi_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# KPI Table — All Scenarios\n\n")
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("|" + "|".join(["---"] * len(fieldnames)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[k]) for k in fieldnames) + " |\n")


def ablation_table(results: dict, output_dir: str) -> None:
    """Write ablation analysis table (doc 7.5) — EXP-5 only."""
    out = pathlib.Path(output_dir)
    if "EXP-5" not in results:
        return
    kpi_table({"EXP-5": results["EXP-5"]}, str(out / "ablation"))


def gram_performance_table(results: dict, output_dir: str) -> None:
    """Write Gram matrix performance table (doc 7.6) — EXP-6 only."""
    out = pathlib.Path(output_dir)
    if "EXP-6" not in results:
        return

    exp6 = results["EXP-6"]
    ctrl_data = exp6["per_controller"].get("full", {})
    seeds = ctrl_data.get("seeds", [])

    rows = []
    for trace in seeds:
        # Inversion count is logged in the controller's gram matrix
        rows.append({
            "seed": trace.get("seed", 0),
            "n_steps": len(trace.get("y", [])),
            "n_filtered": trace.get("n_filtered", 0),
        })

    md_path = out / "gram_performance.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Gram Matrix Performance (EXP-6)\n\n")
        if rows:
            f.write("| seed | n_steps | n_filtered |\n|---|---|---|\n")
            for row in rows:
                f.write(f"| {row['seed']} | {row['n_steps']} | {row['n_filtered']} |\n")
        else:
            f.write("*No data.*\n")


def _fmt(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, dict):
        m = val.get("mean")
        ci_lo = val.get("ci_low")
        ci_hi = val.get("ci_high")
        if m is None:
            return "—"
        if ci_lo is not None and ci_hi is not None:
            return f"{m:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]"
        return f"{m:.4f}"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _nested(kpis: dict, key: str, subkey: str):
    d = kpis.get(key)
    if isinstance(d, dict):
        return d.get(subkey)
    return None
