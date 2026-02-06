# 7) Как воспроизвести и обновить отчёт

## 7.1 Зависимости

Рекомендуемый способ — виртуальное окружение:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -U pip
.\.venv\Scripts\pip install mesa numpy pandas scikit-learn matplotlib pillow pyarrow joblib pypdf
```

## 7.2 Полный прогон (семинарная ветка)

Команды (пример как запускалось для `output_seminar/`):

```powershell
.\.venv\Scripts\python .\scripts\run_lhs.py --n 250 --seed 0 --seeds 0,1,2 --steps 150 --window 30 --balance-ok-threshold 0.6 --format csv --output output_seminar/datasets/lhs_runs.csv
.\.venv\Scripts\python .\scripts\train_emulator.py --data output_seminar/datasets/lhs_runs.csv --targets scripts/targets_report.json --min-r2-for-history-matching 0.25 --outdir output_seminar/results

.\.venv\Scripts\python .\scripts\run_lhs.py --n 250 --seed 1 --seeds 0,1,2 --steps 150 --window 30 --balance-ok-threshold 0.6 --bounds-csv output_seminar/results/refined_intervals.csv --bounds-kind p05p95 --format csv --output output_seminar/datasets/lhs_runs_wave2.csv
.\.venv\Scripts\python .\scripts\train_emulator.py --data output_seminar/datasets/lhs_runs_wave2.csv --targets scripts/targets_report.json --min-r2-for-history-matching 0.25 --outdir output_seminar/results/wave2

.\.venv\Scripts\python .\scripts\make_calibration_summary.py --lhs1 output_seminar/datasets/lhs_runs.csv --hm1 output_seminar/results/history_matching.csv --ri1 output_seminar/results/refined_intervals.csv --lhs2 output_seminar/datasets/lhs_runs_wave2.csv --hm2 output_seminar/results/wave2/history_matching.csv --ri2 output_seminar/results/wave2/refined_intervals.csv --scores1 output_seminar/results/emulator_scores.csv --out output_seminar/results/calibration_summary.md

.\.venv\Scripts\python .\scripts\make_report_set.py --data output_seminar/datasets/lhs_runs_wave2.csv --history-matching output_seminar/results/wave2/history_matching.csv --outdir output_seminar/results/report_set_wave2 --n 6 --steps 300 --window 50 --seeds 0,1,2 --min-employment 55 --min-output 60 --min-consumption 60 --min-hh-deposit 0.2 --max-bank-resolved-share 0.06 --max-bank-bailedout-share 0.02 --balance-ok-threshold 0.98

.\.venv\Scripts\python .\scripts\build_firm_policy_dataset.py --runs-dir output_seminar/results/report_set_wave2 --out output_seminar/ml_datasets/firm_policy.csv
.\.venv\Scripts\python .\scripts\train_firm_policy.py --data output_seminar/ml_datasets/firm_policy.csv --out output_seminar/models/firm_policy.pkl
```

Для запуска Andrianakis-style разложения неопределённостей и SA по EV/OU/MD/CU:

```powershell
.\.venv\Scripts\python .\scripts\train_emulator.py --data output_seminar/datasets/lhs_runs.csv --targets scripts/targets_report.json --min-r2-for-history-matching 0.25 --ev-mode seed_replicates --ev-default-quantile 0.9 --sa-enable --sa-domain nroy --sa-reductions 0.1,0.2,0.3,0.4 --outdir output_seminar/results
```

Новые артефакты:
- `output_seminar/results/uncertainty_components.csv`
- `output_seminar/results/sensitivity_uncertainty.csv`
- `output_seminar/results/sensitivity_report.md`

## 7.4 Full research-core pipeline (single command)

Полный воспроизводимый прогон без baseline-ветки:

```powershell
.\.venv\Scripts\python .\scripts\run_research_core.py --mode full --outdir output_research_core
```

Быстрый dry smoke:

```powershell
.\.venv\Scripts\python .\scripts\run_research_core.py --mode dry --outdir output_research_core_smoke
```

Итоговый master-отчёт:
- `output_research_core/04_master/research_core_report.md`
- таблицы: `output_research_core/04_master/research_core_tables/`

## 7.3 Как обновлять markdown‑отчёт

- Если меняются bounds/таргеты/порог `bad_run`, сначала регенерируй `output_seminar/results/calibration_summary.md`.
- Затем обнови таблицы в:
  - `reports/2026-01-25_seminar/03_results.md` (bounds/LHS/emu/HM/report_set)
  - `reports/2026-01-25_seminar/00_overview.md` (краткие выводы)

Если хочешь, могу добавить небольшой скрипт, который будет автоматически собирать цифры из `output_seminar/*` и генерировать `03_results.md` без ручного редактирования.
