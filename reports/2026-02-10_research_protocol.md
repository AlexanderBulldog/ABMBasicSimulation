# Research Protocol (ABM Calibration v2)

## Scope
- Goal: strict, explainable, reproducible calibration for scientific presentation.
- Fixed model/bounds; only `sigma_model/sigma_obs` are allowed to change between HM iterations.

## Core Formula
- For each target metric:
- `I = |mu - target| / sqrt(var_obs + var_ev + var_md + var_cu)`
- Aggregate:
- `I_max = max_j I_j`
- `NROY = I_max < threshold`

## Required Gates
- `NROY wave2 in [25, 60]%`
- `I_max median wave2 < 3.2`
- Representative long-runs:
- `Employment_mean >= 45`
- `Output_mean >= 50`
- `Consumption_mean >= 50`
- `BankFailed_share == 0`
- `BalanceOK_share == 1`
- Confirmatory:
- `|NROY_confirm - NROY_main| <= 5 pp`
- SA top-2 components are stable.

## Pipeline
1. Wave1 LHS (400x3 in full mode)
2. Wave1 emulator/HM/SA with `sigma_calibration_mode=wave1_empirical`
3. Wave2 LHS in refined intervals
4. Wave2 emulator/HM/SA + iterative sigma-model tuning (max 3 iterations, step 10%)
5. Representative long-runs from NROY wave2 with hard gates
6. Confirmatory rerun with alternative LHS master seed
7. Master report + tables + run config + protocol

## Public Interfaces (v2)
- `scripts/targets_report.json`:
- Supports `_meta` fields: `version_tag`, `estimation_method`, `estimation_sample`.
- `scripts/train_emulator.py`:
- New CLI: `--sigma-calibration-mode {fixed,wave1_empirical}`
- New CLI: `--sigma-calibration-summary PATH`
- New CLI: `--calibrated-targets-out PATH`
- New output: `sigma_calibration_summary.csv`
- `scripts/make_report_set.py`:
- Hard E/O/C floors are enforced in strict validation mode.
- New output: `representative_rejections.csv` with rejection reasons.
- `scripts/make_research_master_report.py`:
- New gate support: representative floors and confirmatory stability.

## Minimal Reproduction Command
```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python .\scripts\run_research_core.py --mode full --outdir output_research_core
```

## Mandatory Outputs
- `output_research_core/run_config.json`
- `output_research_core/04_master/research_protocol.md`
- `output_research_core/04_master/research_core_report.md`
- `output_research_core/04_master/research_core_tables/*.csv`
