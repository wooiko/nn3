"""Unified experiment runner: loads scenario config, runs N seeds, saves traces."""
import json
import pathlib
import numpy as np
import yaml

from object_model.static_model import predict_outputs
from object_model.dynamic_model import DynamicModel
from object_model import disturbances as dist_fns
from baselines.classic_mpc import ClassicMPC
from baselines.prototype_mpc import PrototypeMPC
from mpc_core.mpc_controller import MPCController, MPCControllerNoSVR
from metrics.kpi import compute_all
from metrics.statistics import aggregate_kpis


_CONTROLLERS = {
    "full": MPCController,
    "full_no_svr": MPCControllerNoSVR,
    "classic_mpc": ClassicMPC,
    "prototype_mpc": PrototypeMPC,
}

_SCENARIOS_DIR = pathlib.Path(__file__).parent / "configs"
_OBJ_CFG_PATH = pathlib.Path(__file__).parents[1] / "object_model" / "object_config.yaml"
_GLOBAL_CFG_PATH = pathlib.Path(__file__).parents[1] / "config_global.yaml"


def _load_merged_cfg(scenario_cfg_path: pathlib.Path) -> dict:
    """Merge object_config + config_global + scenario config; scenario has priority."""
    with open(_OBJ_CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(_GLOBAL_CFG_PATH, encoding="utf-8") as f:
        gcfg = yaml.safe_load(f) or {}
    cfg.update({k: v for k, v in gcfg.items() if v is not None})

    with open(scenario_cfg_path, encoding="utf-8") as f:
        scfg = yaml.safe_load(f) or {}
    for k, v in scfg.items():
        if v is not None:
            cfg[k] = v
    return cfg


def _build_disturbance_cfg(cfg: dict, dist_cfg: dict) -> dict:
    """Build a cfg dict with disturbance params overridden from scenario dist block."""
    dcfg = dict(cfg)
    if "magnitude" in dist_cfg:
        dcfg["impulse_magnitude"] = dist_cfg["magnitude"]
        dcfg["step_magnitude"] = dist_cfg["magnitude"]
    if "duration" in dist_cfg:
        dcfg["impulse_duration"] = dist_cfg["duration"]
    if "times" in dist_cfg:
        dcfg["impulse_times"] = dist_cfg["times"]
    if "time" in dist_cfg:
        dcfg["step_time"] = dist_cfg["time"]
    if "rate" in dist_cfg:
        dcfg["drift_rate"] = dist_cfg["rate"]
    if "amplitude" in dist_cfg:
        dcfg["drift_amplitude"] = dist_cfg["amplitude"]
    for key in ("drift_rate", "drift_amplitude", "step_magnitude", "step_time",
                "impulse_magnitude", "impulse_duration", "impulse_times",
                "impulse_times_mixed", "impulse_magnitude_mixed", "impulse_duration_mixed"):
        scn_key = key.replace("_mixed", "")
        if key in dist_cfg:
            dcfg[scn_key] = dist_cfg[key]
    # mixed-specific keys
    for k in ("drift_rate", "drift_amplitude", "step_magnitude", "step_time",
              "impulse_magnitude", "impulse_duration", "impulse_times"):
        if k in dist_cfg:
            dcfg[k] = dist_cfg[k]
    return dcfg


def _get_alpha_disturbance(dist_type: str, k: int, dcfg: dict) -> float:
    if dist_type == "none":
        return 0.0
    elif dist_type == "drift":
        return dist_fns.drift(k, dcfg)
    elif dist_type == "step":
        return dist_fns.step_change(k, dcfg)
    elif dist_type == "impulse":
        return dist_fns.impulse_spike(k, dcfg)
    elif dist_type == "mixed":
        return (dist_fns.drift(k, dcfg)
                + dist_fns.step_change(k, dcfg)
                + dist_fns.impulse_spike(k, dcfg))
    return 0.0


def _run_one_seed(cfg: dict, seed: int, controller_name: str) -> dict:
    """Run one controller for one seed; return trace dict."""
    rng = np.random.default_rng(seed)
    n_steps = int(cfg.get("n_steps", 200))
    noise_enabled = (cfg.get("noise", {}) or {}).get("enabled", True)
    ref_cfg = cfg.get("reference", {}) or {}
    y_ref = np.array([
        float(ref_cfg.get("beta_fe", 0.650)),
        float(ref_cfg.get("epsilon", 0.920)),
    ])
    dist_cfg_block = cfg.get("disturbance", {}) or {}
    dist_type = dist_cfg_block.get("type", "none")
    dcfg = _build_disturbance_cfg(cfg, dist_cfg_block)

    # Build controller
    ctrl_cls = _CONTROLLERS[controller_name]
    ctrl = ctrl_cls(cfg)

    # Build object model
    dyn = DynamicModel(cfg)

    # Initial state
    B0 = (cfg["B_min_dom"] + cfg["B_max_dom"]) / 2.0
    Q0 = (cfg["Q_min_dom"] + cfg["Q_max_dom"]) / 2.0
    rho0 = (cfg["rho_min_dom"] + cfg["rho_max_dom"]) / 2.0
    alpha0 = cfg["alpha_fe_ref"]
    dyn.reset(B0, rho0, Q0, alpha0)

    trace_y: list = []
    trace_u: list = []
    trace_modes: list = []
    trace_sigma2: list = []
    n_filtered = 0

    u_k = np.array([B0, rho0, Q0])

    for k in range(n_steps):
        # Disturbance: additive offset on alpha_fe
        alpha_fe = float(np.clip(
            alpha0 + _get_alpha_disturbance(dist_type, k, dcfg),
            cfg["alpha_fe_min_dom"], cfg["alpha_fe_max_dom"]
        ))

        # Plant step
        y_true = dyn.step(u_k[0], u_k[1], u_k[2], alpha_fe)

        # Measurement noise
        if noise_enabled:
            noise = np.array([
                float(rng.normal(0, cfg.get("noise_std_beta_fe", 0.002))),
                float(rng.normal(0, cfg.get("noise_std_epsilon", 0.003))),
            ])
            y_meas = np.clip(y_true + noise, 0.0, 1.0)
        else:
            y_meas = y_true.copy()

        # Controller step
        if controller_name == "full":
            u_next = ctrl.step(y_meas, y_ref, u_k)
            last_log = ctrl._log[-1] if ctrl._log else {}
            trace_modes.append(last_log.get("mode", "nominal"))
            trace_sigma2.append(last_log.get("sigma2", 0.0))
            if last_log.get("is_anomaly", False):
                n_filtered += 1
        else:
            u_next = ctrl.step(y_meas, y_ref)
            trace_modes.append("nominal")
            trace_sigma2.append(0.0)

        trace_y.append(y_meas.copy())
        trace_u.append(u_k.copy())
        u_k = u_next

    return {
        "controller": controller_name,
        "seed": seed,
        "y": np.array(trace_y),
        "u": np.array(trace_u),
        "modes": trace_modes,
        "sigma2": np.array(trace_sigma2),
        "n_filtered": n_filtered,
        "y_ref": y_ref,
    }


def run_scenario(scenario_id: str, cfg_path: str,
                 results_dir: str | None = None,
                 n_seeds: int | None = None) -> dict:
    """Run all controllers for all seeds; return aggregated results.

    Args:
        scenario_id : e.g. 'EXP-1'
        cfg_path    : relative (from experiments/configs/) or absolute path
        results_dir : if set, saves lightweight summary JSON
        n_seeds     : override n_seeds from config

    Returns:
        {
          'scenario_id': str,
          'cfg': dict,
          'per_controller': {
              '<ctrl>': {'seeds': [...], 'kpis_per_seed': [...], 'kpis': {...}},
          }
        }
    """
    cfg_abs = (_SCENARIOS_DIR / cfg_path
               if not pathlib.Path(cfg_path).is_absolute()
               else pathlib.Path(cfg_path))
    cfg = _load_merged_cfg(cfg_abs)
    cfg["scenario_id"] = scenario_id

    n = n_seeds if n_seeds is not None else int(cfg.get("n_seeds", 1))
    seed_base = int(cfg.get("seed_base", 42))
    controllers = cfg.get("controllers", ["full", "classic_mpc", "prototype_mpc"])
    valid_controllers = [c for c in controllers if c in _CONTROLLERS]

    per_controller: dict = {}
    for ctrl_name in valid_controllers:
        seeds_traces = []
        for i in range(n):
            seed = seed_base + i
            trace = _run_one_seed(cfg, seed, ctrl_name)
            seeds_traces.append(trace)

        y_ref = seeds_traces[0]["y_ref"]
        kpi_list = [compute_all(t, y_ref) for t in seeds_traces]
        per_controller[ctrl_name] = {
            "seeds": seeds_traces,
            "kpis_per_seed": kpi_list,
            "kpis": aggregate_kpis(kpi_list),
        }

    result = {
        "scenario_id": scenario_id,
        "cfg": cfg,
        "per_controller": per_controller,
    }

    if results_dir:
        out_dir = pathlib.Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "scenario_id": scenario_id,
            "per_controller": {
                ctrl: {"kpis": data["kpis"]}
                for ctrl, data in per_controller.items()
            }
        }
        with open(out_dir / f"{scenario_id}_summary.json", "w", encoding="utf-8") as f:
            json.dump(_jsonify(summary), f, indent=2, ensure_ascii=False)

    return result


def _jsonify(obj):
    """Recursively convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj
