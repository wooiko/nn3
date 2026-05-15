"""Entry point: run all EXP-1…EXP-7, generate all tables and figures.

Usage:
    python run_all.py [--config config_global.yaml] [--results-dir results/]

Two identical invocations must produce byte-for-byte identical results (spec 5.7).
"""
import argparse
import hashlib
import json
import pathlib
import yaml

from experiments.scenarios import SCENARIOS
from experiments.exp_runner import run_scenario
from reporting.tables import kpi_table, ablation_table, gram_performance_table
from reporting.figures import transient_response, gp_variance_trajectory, r_vs_sigma2
from reporting.protocol import compile_protocol


def _config_hash(cfg_path: str) -> str:
    """SHA-256 hash of config_global.yaml for traceability (spec 5.5)."""
    with open(cfg_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_global.yaml")
    parser.add_argument("--results-dir", default="results/")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    cfg_hash = _config_hash(args.config)
    print(f"config_global.yaml SHA-256: {cfg_hash}")

    all_results = {}
    for exp_id, cfg_path in SCENARIOS.items():
        print(f"Running {exp_id} …")
        result = run_scenario(exp_id, cfg_path, results_dir=str(results_dir))
        all_results[exp_id] = result

    # Save config hash alongside results
    with open(results_dir / "config_hash.json", "w", encoding="utf-8") as f:
        json.dump({"config_global_sha256": cfg_hash}, f, indent=2)

    print("Generating tables …")
    kpi_table(all_results, str(results_dir))
    ablation_table(all_results, str(results_dir))
    gram_performance_table(all_results, str(results_dir))

    print("Generating figures …")
    for exp_id in SCENARIOS:
        if exp_id in all_results:
            transient_response(all_results, exp_id, str(results_dir))
            gp_variance_trajectory(all_results, exp_id, str(results_dir))
            r_vs_sigma2(all_results, exp_id, str(results_dir))

    print("Compiling protocol …")
    compile_protocol(str(results_dir), str(results_dir / "protocol.md"),
                     all_results=all_results)

    print(f"Done. Results in {results_dir}")


if __name__ == "__main__":
    main()
