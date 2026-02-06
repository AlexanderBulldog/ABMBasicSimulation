# Detailed Research Core Report (ABM -> Emulator/HM -> SA -> Representative Runs)

## 0) Что вы просили и что сделано
Цель: получить прозрачный end-to-end pipeline без baseline:

`ABM (LHS wave1/wave2) -> Emulator + History Matching (Andrianakis-style) -> SA (EV/OU/MD/CU) -> Representative long runs -> научный вывод`

Pipeline реально запущен в режиме `full`:
- конфиг: `output_research_core/run_config.json`
- оркестратор: `scripts/run_research_core.py`
- итоговый тех-отчёт: `output_research_core/04_master/research_core_report.md`

Ниже — подробная, человеко-читаемая интерпретация того, что получилось.

---

## 1) Шаг A: какие данные подали в ABM и как запускали

### 1.1 Параметрическое пространство ABM
Bounds взяты из `scripts/run_operator.py:PARAM_BOUNDS` (13 параметров):
- `alpha_mean`, `alpha_std`
- `wage`, `productivity`
- `loan_rate`, `deposit_rate`
- `bank_credit_multiplier`
- `hh_debt_cap_multiplier`, `firm_debt_cap_multiplier`
- `adaptation_rate`, `price_elasticity`
- `skill_wage_weight`, `demand_smoothing`

### 1.2 Wave1 ABM (не “детский” прогон)
- LHS samples: `n=400`
- seeds per theta: `0,1,2` (то есть 3 реплики на точку)
- steps per run: `180`
- summary window: `30`
- balance filter threshold: `0.60`
- output rows: `1200` (400*3)

Артефакты:
- `output_research_core/01_wave1/lhs_runs.csv`

### 1.3 Wave2 ABM (суженное пространство)
- source bounds: `output_research_core/01_wave1/refined_intervals.csv`
- bounds kind: `p05p95`
- тот же масштаб: `n=400`, seeds `0,1,2`, steps `180`, window `30`
- output rows: `1200`

Артефакты:
- `output_research_core/02_wave2/lhs_runs_wave2.csv`

---

## 2) Шаг B: эмулятор + history matching

### 2.1 Настройки калибровки
Из `output_research_core/run_config.json`:
- targets: `scripts/targets_report.json`
- `min-r2-for-history-matching=0.25`
- `improb-threshold=3.0`
- EV mode: `seed_replicates`
- EV quantile: `0.90`

### 2.2 Результаты качества эмулятора
Из `output_research_core/04_master/research_core_tables/emulator_quality.csv`:
- wave1: trained `22`, skipped `4`, GPR CV mean `0.486`, median `0.591`
- wave2: trained `22`, skipped `4`, GPR CV mean `0.502`, median `0.653`

Интерпретация: качество эмулятора в wave2 чуть лучше, особенно по медиане CV R2.

### 2.3 Результаты HM
Из `output_research_core/04_master/research_core_tables/history_matching_summary.csv`:
- wave1: NROY `74.28%`, `I_max median=2.321`, `p95=4.133`, `max=5.276`
- wave2: NROY `83.50%`, `I_max median=2.158`, `p95=3.703`, `max=4.238`

Интерпретация:
- По `I_max` стало лучше (ниже медиана и хвост).
- Но NROY стал слишком широким (`83.5%`), то есть критерий отсечения оказался мягким.

---

## 3) Шаг C: что сузилось между wave1 и wave2

### 3.1 Глобально по метрикам процесса
Из `output_research_core/04_master/research_core_tables/wave1_wave2_comparison.csv`:
- `NROY_pct`: `74.28 -> 83.50` (увеличение на `+9.22 п.п.`)
- `Imax_median`: `2.321 -> 2.158` (улучшение на `-0.163`)
- `RI_shrink_mean`: `0.79% -> 12.13%` (значимое сужение в wave2)
- `bad_run_pct`: `28.08% -> 25.25%` (умеренное улучшение)

### 3.2 Какие параметры сужаются сильнее (wave2)
Топ shrink из `output_research_core/02_wave2/refined_intervals.csv`:
- `productivity`: ~`19.05%`
- `demand_smoothing`: ~`14.21%`
- `hh_debt_cap_multiplier`: ~`13.92%`
- `wage`: ~`13.84%`
- `loan_rate`: ~`13.26%`
- `alpha_std`: ~`12.36%`
- `adaptation_rate`: ~`11.91%`
- `firm_debt_cap_multiplier`: ~`11.75%`

Интерпретация: процесс начал давать структурное сужение именно в поведенческих и финансово-фрикционных параметрах.

---

## 4) Шаг G: анализ чувствительности EV/OU/MD/CU

Источник: `output_research_core/04_master/research_core_tables/sa_ranking_wave2.csv`

Ранги влияния (чем выше, тем сильнее изменение NROY при сжатии неопределённости):
1. `MD` (model discrepancy) — доминирует
2. `OU` (observation uncertainty)
3. `EV` (ensemble variability)
4. `CU` (code/emulator uncertainty)

Количественно (share@max_reduction=40%):
- `MD`: `0.1602`
- `OU`: `0.0374`
- `EV`: `0.0093`
- `CU`: `0.0067`

Интерпретация:
- Основной ограничитель научной “жёсткости” сейчас — не эмулятор, а предположения о discrepancy (MD).
- Улучшения эмулятора полезны, но не дадут главного эффекта без ревизии `sigma_model`/целевых допусков.

---

## 5) Репрезентативные длинные прогоны (не smoke)

Настройки:
- points: `n=8`
- seeds: `0,1,2`
- long horizon steps: `400`
- window: `60`
- total long runs: `24`

Артефакты:
- `output_research_core/03_report_set/representative_points.csv`
- `output_research_core/03_report_set/representative_summary.csv`
- `output_research_core/03_report_set/rep_plots/rep_00.png ... rep_07.png`

Факт по устойчивости:
- `BankFailed_share max = 0`
- `BalanceOK_share min = 1`
- Коллапса в строгом смысле нет (все макроагрегаты >0).

Но важный нюанс качества:
- есть слабые режимы (напр. `rep_id=5`, `rep_id=6`) с низкими средними:
  - Employment_mean до `10`
  - Output_mean до `15.12`
  - Consumption_mean до `12.51`

Интерпретация:
- Pipeline технически стабилен.
- Однако “репрезентативность” части long-runs пока слабая для сильного нарратива статьи.

---

## 6) Строгий quality gate: итог

Из `output_research_core/04_master/research_core_tables/quality_gates.csv`:
- PASS: 9/10 условий
- FAIL: `NROY wave2 in [25,60]%` (получено `83.50%`)

Итог:
- Pipeline **работает корректно end-to-end**.
- Научное ядро **ещё не “сильное” по жёсткости калибровки**, потому что область NROY слишком широкая.

---

## 7) Что означает цепочка A -> B -> V -> G на ваших данных

- **A (данные из ABM):** wave1/wave2 LHS по 1200 строк, хорошие реплики по seed.
- **B (эмулятор/HM):** эмулятор обучается стабильно, HM считает I_max корректно и улучшает форму распределения I_max.
- **V (улучшение):**
  - есть сужение параметров (mean shrink wave2 ~12.13%),
  - есть улучшение `I_max` (медиана ниже),
  - есть снижение bad_run.
- **G (интерпретация):**
  - главный рычаг — `MD`,
  - потом `OU`,
  - `EV/CU` вторичны.

Практический вывод: смысл pipeline подтверждён, но для “сильного” ядра нужен ещё один цикл ужесточения uncertainty assumptions/targets.

---

## 8) Рекомендованный следующий шаг (чтобы довести до сильного ядра)

1. Сделать отдельный run с более строгим uncertainty-профилем:
- уменьшить `sigma_model` для ключевых метрик (пошагово: -10%, -20%),
- проверить влияние на NROY и качество report-set.

2. Поднять минимальные фильтры representative set:
- добавить минимумы не только >0, но и ближе к макро-реалистичным порогам (например Employment_mean >= 45, Output_mean >= 50, Consumption_mean >= 50 для всех seed).

3. Повторить wave2 + report_set и проверить строгий gate до PASS.

База (`baseline`) в этом ядре действительно не нужна и корректно исключена.
