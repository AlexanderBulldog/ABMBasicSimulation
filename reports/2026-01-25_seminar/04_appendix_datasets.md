# 4) Приложение: датасеты, колонки, “что где лежит”

## 4.1 LHS датасеты (основной формат)

Файлы:

- baseline wave1: `output/datasets/lhs_runs.csv`
- baseline wave2: `output/datasets/lhs_runs_wave2.csv`
- seminar wave1: `output_seminar/datasets/lhs_runs.csv`
- seminar wave2: `output_seminar/datasets/lhs_runs_wave2.csv`

Размерность:

- baseline: 600 строк (200 LHS × 3 seed)
- seminar: 750 строк (250 LHS × 3 seed)

### 4.1.1 Колонки параметров (input)

Всего 13:

- `alpha_mean`, `alpha_std`
- `wage`, `productivity`
- `loan_rate`, `deposit_rate`
- `bank_credit_multiplier`
- `hh_debt_cap_multiplier`, `firm_debt_cap_multiplier`
- `adaptation_rate`
- `price_elasticity`
- `skill_wage_weight`
- `demand_smoothing`

### 4.1.2 Колонки метрик (output)

Всего 26 (семантика: средние по последнему окну + доли/ставки):

- `Employment_mean`, `UnemploymentRate_mean`
- `Output_mean`, `Consumption_mean`, `Transfers_mean`, `AvgPrice_mean`
- `HH_Debt_mean`, `HH_Deposit_mean`, `Firm_Debt_mean`, `Bank_Equity_mean`
- `BankFailed_mean`, `BankResolved_mean`, `BankResolutionAmount_mean`, `BankResolutionHaircut_mean`
- `BankBailedOut_mean`, `BankBailoutAmount_mean`
- `Defaults_mean`, `DefaultsHH_mean`, `DefaultsFirm_mean`, `BalanceOK_mean`
- `DefaultsHH_rate`, `DefaultsFirm_rate`
- `BalanceOK_share`
- `BankFailed_share`, `BankResolved_share`, `BankBailedOut_share`

### 4.1.3 Служебные колонки

- `seed` — повторяемость стохастики
- `bad_run` — флаг “неиспользуемого/патологического” прогона (в seminar‑ветке)
- `error` — строка ошибки, если прогон упал

## 4.2 Эмулятор (качество)

Файлы:

- `output_seminar/results/emulator_scores.csv`
- `output_seminar/results/wave2/emulator_scores.csv`

Содержит для каждой метрики:

- `gpr_r2`, `gpr_rmse` (holdout)
- `gpr_r2_cv_mean/std` (KFold)
- аналоги для `rf_*`
- `y_transform` (identity / log1p / signed_log1p)
- `skipped` (если метрика почти константная)

## 4.3 History matching (NROY)

Файлы:

- `output_seminar/results/history_matching.csv`
- `output_seminar/results/wave2/history_matching.csv`

Содержит:

- параметры `theta`
- по каждой таргет‑метрике: `*_pred`, `*_sigma_emul`, `*_I`
- `I_max` и булевый флаг `nroy`

Таргеты:

- `scripts/targets_report.json`

## 4.4 Refined intervals

Файлы:

- `output_seminar/results/refined_intervals.csv`
- `output_seminar/results/wave2/refined_intervals.csv`

Колонки:

- `initial_low/high` — исходные bounds (априорные)
- `nroy_min/max`, `nroy_p05/p95` — статистики внутри NROY
- `shrink_pct` — сжатие интервала по min/max относительно initial

## 4.5 Report set (репрезентативные траектории)

Файлы:

- `output_seminar/results/report_set_wave2/representative_points.csv`
- `output_seminar/results/report_set_wave2/representative_summary.csv`
- `output_seminar/results/report_set_wave2/rep_run_XX_seedY.csv`

`representative_points.csv` — выбранные точки параметров (и служебные поля отбора).  
`representative_summary.csv` — агрегированные метрики по каждому rep_id и seed.  
`rep_run_*.csv` — временные ряды агрегатов по шагам (для графиков в презентацию).

## 4.6 ML‑политика фирм

Файлы:

- датасет: `output_seminar/ml_datasets/firm_policy.csv`
- модель: `output_seminar/models/firm_policy.pkl`

Схема датасета:

- features: `inventory`, `last_demand`, `cash`, `debt`, `price`, `n_workers`, `bank_available_credit`, `wage`
- target: `realized_markup`
- keys: `step`, `rep_id`, `seed`, `firm_id`

