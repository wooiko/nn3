"""Reproducibility tests: two identical runs must produce identical results (spec 5.7)."""
import numpy as np
import pytest
from experiments.exp_runner import run_scenario


def test_double_run_identical():
    """Two runs of EXP-1 (n_seeds=1) must produce byte-for-byte identical y, u arrays."""
    r1 = run_scenario("EXP-1", "exp1_nominal.yaml", n_seeds=1)
    r2 = run_scenario("EXP-1", "exp1_nominal.yaml", n_seeds=1)

    for ctrl_name in r1["per_controller"]:
        seeds1 = r1["per_controller"][ctrl_name]["seeds"]
        seeds2 = r2["per_controller"][ctrl_name]["seeds"]
        assert len(seeds1) == len(seeds2)
        for t1, t2 in zip(seeds1, seeds2):
            y1 = np.array(t1["y"])
            y2 = np.array(t2["y"])
            u1 = np.array(t1["u"])
            u2 = np.array(t2["u"])
            assert np.array_equal(y1, y2), (
                f"Controller {ctrl_name}: y trajectories differ between runs\n"
                f"Max diff: {np.abs(y1 - y2).max()}"
            )
            assert np.array_equal(u1, u2), (
                f"Controller {ctrl_name}: u trajectories differ between runs"
            )
