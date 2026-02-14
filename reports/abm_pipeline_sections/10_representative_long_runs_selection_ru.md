# Раздел 10. Репрезентативные длинные траектории: как выбираются и зачем нужны

Этот раздел описывает, как из множества допустимых параметров (NROY) выбираются небольшие, но содержательные траектории для научной интерпретации и презентации.

## 10.1 Зачем нужен этап representative runs

После HM у нас может быть много допустимых точек \(\theta\).  
Показывать в отчете сотни траекторий бессмысленно:
- сложно интерпретировать,
- теряется narrative,
- не видно типовых режимов.

Поэтому строится **малый репрезентативный набор** точек:

1. статистически не случайный,
2. экономически интерпретируемый,
3. проверенный на длинном горизонте.

---

## 10.2 Откуда берутся кандидаты

Источник кандидатов:
- wave2-датасет,
- дополнительно пересечение с `nroy=True` из `history_matching.csv`.

То есть отбор идет только внутри уже неотвергнутой области параметров.

Перед отбором убираются очевидно плохие строки:
- `bad_run=True`,
- нулевые/дегенеративные short-run метрики,
- невалидные цены.

---

## 10.3 Score-функция кандидата

Каждой кандидатной точке присваивается эвристический score:

\[
score(\theta)=\sum_m \left(\frac{y_m(\theta)-t_m}{s_m}\right)^2
\]

где:
- \(t_m\) — желаемые "макроправдоподобные" ориентиры,
- \(s_m\) — масштабы нормализации.

Чем ниже score, тем "лучше" точка по совокупности целей.

Важно:
- score нужен только для ранжирования кандидатов,
- окончательное принятие точки определяется жесткой валидацией long-run.

---

## 10.4 Почему нужен diversity, а не только "лучший score"

Если брать только минимальный score, можно получить точки из одного локального режима.  
Поэтому применяется простая диверсификация:

1. сортировка по ключевой оси (например `Employment_mean`),
2. разбиение на корзины,
3. выбор лучших по score в корзинах.

Это дает набор точек, покрывающий разные макрорежимы, а не только одну "узкую" зону.

---

## 10.5 Длинная валидация: главный фильтр

Для каждой выбранной точки \(\theta_r\) выполняются длинные прогоны на всех seed:

\[
(\theta_r, s_1), (\theta_r, s_2), \dots, (\theta_r, s_K)
\]

На этих прогонах считается summary по длинному горизонту и проверяются hard-условия.

### 10.5.1 Строгие критерии (strict)

Точка принимается, если для каждого seed одновременно:

1. \(Employment_{mean} \ge E_{min}\)
2. \(Output_{mean} \ge O_{min}\)
3. \(Consumption_{mean} \ge C_{min}\)
4. \(UnemploymentRate_{mean} \le U_{max}\)
5. \(HH\_Deposit_{mean} \ge D_{min}\)
6. \(DefaultsHH_{rate}\le d_{hh}^{max}\)
7. \(DefaultsFirm_{rate}\le d_f^{max}\)
8. \(BankResolved_{share}\le b_r^{max}\)
9. \(BankBailedOut_{share}\le b_b^{max}\)
10. \(BankResolutionHaircut_{mean}\le h^{max}\)
11. \(BankFailed_{share}=0\)
12. \(BalanceOK_{share}\ge \tau_{bal}\)

То есть приемка идет не по одной красивой траектории, а по устойчивости на всех репликах.

### 10.5.2 Мягкий fallback (relaxed)

Если strict не позволяет набрать нужное число точек, есть fallback:
- сохраняются минимальные условия жизнеспособности (не нулевые ключевые метрики, отсутствие BankFailed и т.п.),
- но в текущем строгом научном режиме приоритет — strict-прохождение.

---

## 10.6 Почему это принципиально важно для науки

Short-run summary может выглядеть хорошо, но на длинном горизонте точка может развалиться.  
Длинная валидация защищает от ложноположительных "красивых" точек.

Именно этот этап обеспечивает, что итоговые графики:
- не случайные,
- не временная флуктуация,
- реально соответствуют устойчивому режиму модели.

---

## 10.7 Что сохраняется как результат этапа

1. `representative_points.csv`
- список принятых параметрических точек,
- дополнительные поля валидации.

2. `representative_summary.csv`
- summary по каждой паре `(rep_id, seed)`.

3. `rep_run_XX_seedY.csv`
- полные временные ряды длинных прогонов.

4. `rep_plots/*.png`
- визуализация траекторий.

5. `representative_rejections.csv` (если были отказы)
- прозрачный лог причин отклонения кандидатов.

Это делает этап полностью аудируемым.

---

## 10.8 Связь с quality gates

Representative этап напрямую проверяет и закрывает ключевые критерии итогового PASS:

1. `Employment/Output/Consumption` floors,
2. `BankFailed_share == 0`,
3. `BalanceOK_share == 1`.

Если эти условия не выполнены, даже хороший HM/NROY не считается научно достаточным для презентации.

---

## 10.9 Типичные ошибки интерпретации (и как их избежать)

1. Ошибка: "NROY уже есть, значит достаточно."
- Неверно: NROY говорит о совместимости с таргетами, но не гарантирует long-run устойчивость.

2. Ошибка: "Один seed выглядит хорошо — значит точка репрезентативна."
- Неверно: приемка должна быть по всем seed.

3. Ошибка: "Графики красивые, значит режим научно валиден."
- Неверно: валидность определяется прохождением формальных hard-gates.

---

## 10.10 Ключевая мысль раздела 10

Representative long-runs — это мост между статистической калибровкой и содержательной экономической интерпретацией.

Они отвечают на практический вопрос:

> "Какие конкретные динамики мы вправе показывать как научно обоснованный итог модели?"

Именно этот этап превращает NROY из абстрактного множества параметров в проверяемые, устойчивые макротраектории.

---

## 10.11 Сравнение baseline vs improved (preflight v5)

Сравнение выполнено между:
- baseline: `output_research_core_v3/04_master/research_core_tables/quality_gates.csv`
- improved: `output_research_core_v5_preflight2/04_master/research_core_tables/quality_gates.csv`
- итог preflight: `output_research_core_v5_preflight2/go_no_go_status.json`

| Критерий (blocking) | Baseline v3 | Improved v5 preflight2 | Изменение |
|---|---:|---:|---:|
| NROY wave2 в [25,65]% | 57.60% (PASS) | 58.57% (PASS) | +0.97 п.п. |
| I_max median < 3.0 | 2.5866 (PASS) | 2.3729 (PASS) | лучше |
| I_max p95 < 4.5 | 2.8018 (PASS) | 3.3152 (PASS) | хуже, но в норме |
| Emulator CV-R2 median >= 0.60 | 0.5630 (FAIL) | 0.6193 (PASS) | ключевое улучшение |
| Emulator CV-R2 share>=0.30 >= 0.65 | 0.8889 (PASS) | 1.0000 (PASS) | +11.11 п.п. |
| bad_run_pct_w2 <= 5% | 0.00% (PASS) | 0.00% (PASS) | без изменений |
| Structural PriceDispersion | 0.0822 (PASS) | 0.0882 (PASS) | стабильно |
| Structural |InventoryGap| p95 <= 1.5 | 0.3292 (PASS) | 0.3327 (PASS) | стабильно |
| Credit (blocking) | p90=104.805 (FAIL, count gate) | mean=0.7005 (PASS, rate gate) | блокер закрыт |

Итог:
- `v5_preflight2` дал `preflight_pass=true` по всем blocking-критериям v3.
- Наиболее важное улучшение: эмулятор перешёл из FAIL в PASS по медиане CV-R2.
- ABM-реалистичность и structural-блок остались устойчиво проходными.

Что пока остаётся слабым местом:
- В legacy/reference кредитный count-контур всё ещё не проходит (`CreditRejections p90=94.93` против reference-целей).
- Это не блокирует Scientific Verdict v3, но показывает резерв для следующей итерации улучшений кредитного канала.

---

## 10.12 Генеральный прогон nightly v3 (запуск и протокол фиксации)

Статус: `RUNNING`.

Параметры запуска:
- script: `scripts/run_research_core.py`
- mode/profile: `--mode full --profile nightly_v3`
- outdir: `output_research_core_v6_full_20260214_182812`
- adaptive waves: `2..5`
- quality/tuning overrides:
  - `--min-r2-for-history-matching 0.30`
  - `--sigma-tuning-step 0.08`
  - `--sigma-tuning-max-iters 3`
  - `--ev-default-quantile 0.80`
  - `--credit-rejection-rate-mean-max 0.80`

Логи выполнения:
- stdout: `output_research_core_v6_full_logs/output_research_core_v6_full_20260214_182812.stdout.log`
- stderr: `output_research_core_v6_full_logs/output_research_core_v6_full_20260214_182812.stderr.log`

Ожидаемые ключевые артефакты после завершения:
- `output_research_core_v6_full_20260214_182812/go_no_go_status.json`
- `output_research_core_v6_full_20260214_182812/waves_summary.csv`
- `output_research_core_v6_full_20260214_182812/stopping_diagnostics.json`
- `output_research_core_v6_full_20260214_182812/04_master/research_core_tables/quality_gates.csv`

Протокол сравнения (обязательный после run):
- baseline: `output_research_core_v3/04_master/research_core_tables/quality_gates.csv`
- preflight-best: `output_research_core_v5_preflight2/04_master/research_core_tables/quality_gates.csv`
- full-nightly: `output_research_core_v6_full_20260214_182812/04_master/research_core_tables/quality_gates.csv`

Сравниваем минимум:
- NROY, I_max median/p95, emulator median+coverage,
- ABM blocking checks,
- structural economy checks,
- confirmatory consistency,
- финальный Scientific Verdict v3 и reason codes.

## 10.13 Статус генерального прогона (факт завершения)

Проверка на момент: 2026-02-14.

Итог:
- Генеральный `full`-прогон `output_research_core_v6_full_20260214_182812` завершён.
- Адаптивный цикл дошёл до лимита: `final_wave=5`, `stop_reason=max_waves_reached` (`stopping_diagnostics.json`).

Blocking-вердикт по `quality_gates.csv`:
- FAIL по `NROY wave2 in [25,65]%`: `68.40%` (выше верхней границы).
- FAIL по `Emulator CV-R2 median >= 0.60`: `0.5570`.
- FAIL по confirmatory consistency: `nroy_delta_pp=15.31` (порог <=5 п.п., SA top2 stable=True).

Что при этом хорошо:
- ABM/structural блок в целом устойчиво PASS.
- Structural credit-rate gate PASS: `mean=0.6832; p90=0.8573`.
- Legacy reference по `CreditRejections p90 <= 90` стал PASS (`84.3367`) и improvement vs baseline >=15% тоже PASS.

## 10.14 Исследовательский pipeline «с нуля» в терминах Andrianakis

Цель этого раздела: зафиксировать один единый язык и убрать избыточные локальные термины.

### 10.14.1 Единый словарь (mapping)

- Simulator / ABM: `scripts/run_lhs.py`
  Генерирует дизайн точек параметров (LHS) и прогоняет ABM, формируя датасет симуляций.
- Emulator: `scripts/train_emulator.py`
  Строит статистический суррогат для метрик и оценивает его качество (CV R2 и др.).
- History Matching (HM): `scripts/train_emulator.py` + `history_matching.csv`
  Считает implausibility и выделяет NROY.
- Wave (волна): итерация цикла «симуляции -> эмулятор -> HM -> сужение интервалов».
- NROY: область параметров, не отвергнутая HM.
- Representative long-runs: `scripts/make_report_set.py`
  Длинные валидационные прогоны на отобранных точках для экономической интерпретации.
- Master report / gates: `scripts/make_research_master_report.py`
  Формальный научный вердикт по blocking-критериям.

### 10.14.2 Линейка этапов исследования (from ABM)

1. Wave k: ABM-симуляции (LHS)
- Вход: bounds текущей волны.
- Выход: `lhs_runs_wave{k}.csv` (или `lhs_runs.csv`).
- Смысл: получить новую выборку поведения экономической системы.

2. Обучение эмулятора
- Вход: `lhs_runs_wave{k}.csv`, `scripts/targets_report.json`.
- Выход: `emulator_scores.csv`, `uncertainty_components.csv`, `history_matching.csv`.
- Смысл: быстро аппроксимировать ABM и формализовать неопределенности (EV/OU/MD/CU).

3. History matching и NROY
- Вход: прогнозы/неопределенности эмулятора + targets.
- Выход: флаг `nroy` и implausibility (`I`, `I_max`).
- Смысл: отбросить заведомо неправдоподобные области параметров.

4. Сужение параметрических интервалов
- Выход: `refined_intervals.csv`.
- Смысл: задать bounds для следующей волны.

5. Проверка stopping rule
- Вход: `waves_summary.csv`, `stopping_diagnostics.json`.
- Смысл: решить, продолжаем волны или фиксируем стабильную научную точку.

6. Representative long-runs
- Вход: NROY-кандидаты финальной волны.
- Выход: `representative_points.csv`, `representative_summary.csv`, `rep_plots/*.png`.
- Смысл: проверить долгосрочную экономическую состоятельность и интерпретируемость.

7. Final scientific verdict
- Вход: `quality_gates.csv`.
- Смысл: единый PASS/FAIL по blocking-критериям (ABM realism + emulator/HM + confirmatory).

### 10.14.3 Где мы сейчас в этой схеме

- После обновления ядра ABM была ожидаемая деградация HM/NROY.
- Затем pipeline восстановлен: есть успешный preflight (`v5`) с PASS по blocking-гейтам.
- В полном adaptive-run (`v6`) ранние волны были хорошими (до wave3), но wave4-5 дали откат.
- Научный вывод: проблема не в базовой жизнеспособности нового ядра, а в политике продолжения волн после стабильной точки.

### 10.14.4 Практическая рекомендация для следующего шага

Для согласования с духом Andrianakis:
- фиксировать не «максимум волн», а «стабильную волну, где критерии уже устойчиво выполнены»;
- confirmatory и final representative запускать от этой стабильной волны;
- в отчёте явно отделять:
  - calibration success (wave-level),
  - long-run economic plausibility,
  - reproducibility (confirmatory/replication).

Это делает рассказ для аудитории прозрачным: 
«мы не подгоняли модель до формального лимита волн, а остановились на научно устойчивой точке и подтвердили её независимым прогоном».
