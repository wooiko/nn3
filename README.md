# nn3 — Adaptive MPC with Kernel Models for Magnetic Separation

**Type:** Model-in-the-Loop (MIL) research verification code  
**Object:** Wet drum magnetic separator PBM 90/250 (GOST 10512-78)  
**Stack:** Python · numpy · scipy · scikit-learn · cvxpy  
**Controller:** Adaptive MPC on a functionally distributed kernel model (SVR/KRR/GP)

---

## Setup

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Verify the object model (should show 10 passed)
pytest tests/integration/test_object_model.py -v

# 3. Run all unit and integration tests (should show 31 passed)
pytest tests/ -v
```

---

## Reproduction

```bash
# Run all EXP-1…EXP-7 and generate all tables, figures, and the summary protocol
python run_all.py

# Results are written to results/
#   kpi_table.csv / kpi_table.md   — doc 7.1 (KPI comparison table)
#   ablation/kpi_table.md          — doc 7.5 (ablation analysis)
#   gram_performance.md            — doc 7.6 (Gram matrix benchmark)
#   EXP-*_transient.png            — doc 7.2 (transient responses)
#   EXP-*_gp_variance.png          — doc 7.3 (σ²_GP trajectories)
#   EXP-*_r_vs_sigma2.png          — doc 7.4 (R(k) vs σ²_GP)
#   protocol.md                    — doc 7.8 (summary protocol)
#   config_hash.json               — config traceability hash (spec 5.5)
```

Two identical `python run_all.py` invocations produce byte-for-byte identical results.

---

## Project Structure

```
object_model/       Object model (static, dynamic, disturbances, validation)
kernel_models/      Gram matrix, SVR filter, KRR model, GP supervisor
mpc_core/           Predictor, R-adaptation, QP solver, MPC controller
baselines/          Classic MPC (fixed R₀), Prototype MPC (autocorrelation R)
experiments/        Experiment runner, scenarios, configs/exp1…exp7.yaml
metrics/            KPI calculations, confidence-interval statistics
reporting/          Tables (CSV/MD), figures (PNG), summary protocol
tests/
  unit/             SVR filter, KRR, GP supervisor, QP solver
  integration/      Full MPC step, object model validation protocol
  regression/       Reproducibility (two runs → identical results)
docs/               Technical spec, object spec, model specs, execution plan
config_global.yaml  Single source of all global numeric parameters
run_all.py          Single entry point for EXP-1…EXP-7
```

---

## Key Architectural Rules

| Rule | Description |
|------|-------------|
| **Single Gram matrix** | `gram_matrix.py` computes `(K+λI)⁻¹` once per step; KRR and GP read from it |
| **Single parameter source** | Every numeric parameter lives in `config_global.yaml` or `object_model/object_config.yaml` with a source comment |
| **Single control range source** | `docs/object_spec_PBM_90_250.md` (GOST 10512-78) — authoritative for B, ρ, Q ranges |
| **Reproducibility** | All RNGs seeded from config; two runs of `run_all.py` produce identical results |

---

## Experiments

| ID | Scenario | Purpose |
|----|----------|---------|
| EXP-1 | Nominal operation, no disturbances | Baseline functionality |
| EXP-2 | Impulse measurement anomalies | SVR filter effectiveness |
| EXP-3 | Domain drift (αFe) | GP supervisor → conservative mode |
| EXP-4 | Step disturbance in feed | Transient quality under adaptive R(k) |
| EXP-5 | Ablation: full vs no-SVR vs fixed-R | Component contribution |
| EXP-6 | Shared Gram matrix vs separate inversions | Computational efficiency |
| EXP-7 | Long mixed-disturbance run | Integral assessment |

---

## Model Status

The object model is **phenomenological**: structure is physically motivated
(Rayner & Napier-Munn 2003; GOST 10512-78). Coefficients are free identification
parameters. The model is **suitable for relative controller comparison** on an
identical simulation plant; it is **not** a quantitatively validated model of
a specific production unit.

See `docs/validation_report_object_model.md` for the full validation protocol (10/10 checks passed).
