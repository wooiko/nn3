# CLAUDE.md — nn3 Project

## Початок кожної сесії

1. Прочитати [`docs/postmortem.md`](docs/postmortem.md) — активні помилки, яких треба уникати.
2. Прочитати [`docs/execution_plan.md`](docs/execution_plan.md) — поточний статус етапів.
3. Продовжувати з першого незавершеного завдання.

---

## Проект

**Тип:** дослідницький верифікаційний код, Model-in-the-Loop (MIL)
**Об'єкт:** магнітний сепаратор ПБМ 90/250
**Стек:** Python · numpy · scipy · scikit-learn · cvxpy
**Регулятор:** адаптивний MPC на функціонально розподіленій ядерній моделі SVR/KRR/GP

---

## Структура

```
object_model/       — модель об'єкта (статика, динаміка, збурення, валідація)
kernel_models/      — gram_matrix, svr_filter, krr_model, gp_supervisor
mpc_core/           — predictor, r_adaptation, qp_solver, mpc_controller
baselines/          — classic_mpc (фіксована R₀), prototype_mpc (автокореляція)
experiments/        — exp_runner, scenarios, configs/exp1…exp7.yaml
metrics/            — kpi, statistics
reporting/          — tables, figures, protocol
tests/
  unit/             — svr_filter, krr_model, gp_supervisor, qp_solver
  integration/      — mpc_step, object_model
  regression/       — reproducibility
docs/               — технічне завдання, специфікації, план, постмортем
config_global.yaml  — єдине джерело всіх числових параметрів
run_all.py          — єдина точка запуску EXP-1…EXP-7
```

---

## Ключові архітектурні правила

- **Єдина матриця Грама.** `gram_matrix.py` обчислює `(K+λI)⁻¹` один раз за крок. KRR і GP читають результат звідти — самостійно не інвертують.
- **Єдине джерело параметрів.** Кожен числовий параметр живе в `config_global.yaml` або `object_model/object_config.yaml` з коментарем-джерелом у форматі `# source: [автор, рік, формула/таблиця/стор.]`. Параметри без джерела — заборонені в production-коді.
- **Єдине джерело діапазонів керування.** `docs/object_spec_PBM_90_250.md` — пріоритет над будь-якими іншими значеннями B, ρ, Q.
- **Відтворюваність.** Усі ГВЧ ініціалізуються seed з конфігу. Два запуски `run_all.py` мають давати ідентичні результати.

---

## Правила коду

- Всі модулі зараз — заготовки з `raise NotImplementedError`. Реалізовувати поетапно згідно з планом (Е1 → Е3/Е2 → Е4 → Е5 → Е6 → Е7).
- Тести писати паралельно з реалізацією, не після.
- Не змінювати діапазони B/ρ/Q без звірки з `object_spec_PBM_90_250.md`.
- Параметри `θ₁, θ₂, R₀, α, γ` для R-адаптації — конфігуровані, не хардкодити.
- `qp_solver.py` — перевіряти feasibility на кожному кроці; нефeasible задача є помилкою, не виключенням.

---

## Запуск

```bash
# Встановити залежності
pip install -r requirements.txt

# Запустити всі експерименти
python run_all.py

# Запустити тести
pytest tests/
```

---

## Документи проекту

| Файл | Призначення |
|------|-------------|
| `docs/technical_spec_dev.md` | Технічне завдання (джерело істини для вимог) |
| `docs/object_spec_PBM_90_250.md` | Специфікація об'єкта — діапазони B, ρ, Q за ГОСТ |
| `docs/static_model_spec.md` | Специфікація статичної моделі |
| `docs/execution_plan.md` | План виконання з чекбоксами по етапах |
| `docs/postmortem.md` | Журнал помилок розробки (active / resolved / dismissed) |
