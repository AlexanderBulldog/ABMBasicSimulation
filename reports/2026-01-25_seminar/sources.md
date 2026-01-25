# Sources / Артефакты (что использовано в отчёте)

## Семинарные результаты

- `output_seminar/results/calibration_summary.md`
- `output_seminar/datasets/lhs_runs.csv`
- `output_seminar/results/emulator_scores.csv`
- `output_seminar/results/history_matching.csv`
- `output_seminar/results/refined_intervals.csv`
- `output_seminar/datasets/lhs_runs_wave2.csv`
- `output_seminar/results/wave2/emulator_scores.csv`
- `output_seminar/results/wave2/history_matching.csv`
- `output_seminar/results/wave2/refined_intervals.csv`
- `output_seminar/results/report_set_wave2/representative_points.csv`
- `output_seminar/results/report_set_wave2/representative_summary.csv`
- `output_seminar/results/report_set_wave2/rep_run_00_seed0.csv` (и далее `rep_run_*.csv`)
- `output_seminar/ml_datasets/firm_policy.csv`
- `output_seminar/models/firm_policy.pkl`

## Baseline результаты (для сравнения)

- `output/results/calibration_summary_newformat.md`
- `output/datasets/lhs_runs.csv`
- `output/results/emulator_scores.csv`
- `output/results/history_matching.csv`
- `output/results/refined_intervals.csv`
- `output/datasets/lhs_runs_wave2.csv`
- `output/results/wave2/emulator_scores.csv`
- `output/results/wave2/history_matching.csv`
- `output/results/wave2/refined_intervals.csv`
- `output/results/report_set/representative_summary.csv`

## Код модели и пайплайна (ссылки)

- bounds и оператор прогонов: `scripts/run_operator.py:21`
- LHS: `scripts/run_lhs.py`
- эмулятор + HM: `scripts/train_emulator.py`
- report set: `scripts/make_report_set.py`
- ML‑датасет: `scripts/build_firm_policy_dataset.py`
- ML‑модель: `scripts/train_firm_policy.py`
- ABM ядро: `src/synpop/model.py`
- агенты: `src/synpop/agents.py`
- банк: `src/synpop/bank.py`
- метрики/summary: `src/synpop/scenarios.py`
- таргеты: `scripts/targets_report.json`

