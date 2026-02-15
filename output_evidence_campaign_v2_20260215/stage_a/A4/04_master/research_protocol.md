# Research Core Protocol

- version_tag: `research-core-v3`
- created_utc: `2026-02-14T23:18:40.091419+00:00`
- goal: `economically_explainable_strict_calibration_for_scientific_presentation`
- allowed_adjustments: `['sigma_model', 'sigma_obs']`

## HM Formula
- `I = |mu - target| / sqrt(var_obs + var_ev + var_md + var_cu)`
- `I_max = max_j I_j`, `NROY = (I_max < threshold)`

## Quality Gates (v3)
- Active NROY band: [25.0, 65.0]%
- I_max median < 3.0
- I_max p95 < 4.5
- Emulator median CV-R2 >= 0.6
