# Research Core Protocol

- version_tag: `research-core-v2`
- created_utc: `2026-02-10T11:56:02.109898+00:00`
- goal: `strict_calibration_for_scientific_presentation`
- allowed_adjustments: `['sigma_model', 'sigma_obs']`

## HM Formula
- `I = |mu - target| / sqrt(var_obs + var_ev + var_md + var_cu)`
- `I_max = max_j I_j`, `NROY = (I_max < threshold)`

## Quality Gates
- NROY wave2 in [25.0, 60.0]%
- I_max median wave2 < 3.2
- Representative floors: E>=45.0, O>=50.0, C>=50.0
