# Раздел 11. Confirmatory-run: независимая проверка устойчивости вывода

Этот раздел описывает, как после выбора репрезентативных длинных траекторий (раздел 10) проверяется, что итог не является случайным артефактом конкретного LHS-дизайна.

## 11.1 Зачем нужен confirmatory-run

Даже при хорошем main-cycle возможен риск, что результат сильно зависит от конкретной выборки параметров в LHS.

Confirmatory-run нужен, чтобы ответить на вопрос:

> "Сохранится ли научный вывод, если повторить pipeline на независимом дизайне с теми же правилами?"

То есть это тест **воспроизводимости вывода**, а не дополнительная подгонка модели.

---

## 11.2 Что именно считается независимым

В текущей реализации независимость достигается за счет сдвига master-seed для LHS:

\[
seed_{w1}^{confirm}=seed_{w1}^{main}+seed\_offset,
\quad
seed_{w2}^{confirm}=seed_{w2}^{main}+seed\_offset
\]

где по умолчанию `seed_offset = 101`.

При этом сохраняются неизменными:
- ABM-ядро и bounds,
- HM/SA формулы,
- quality gate thresholds,
- логика sigma-calibration/tuning.

Такой дизайн проверяет именно устойчивость результата к альтернативной параметрической выборке.

---

## 11.3 Как строится confirmatory-ветка в пайплайне

В `scripts/run_research_core.py` confirmatory выполняется как отдельный mini-pipeline:

1. `05_confirmatory/01_wave1`:
- LHS wave1 с offset-seed,
- обучение эмулятора,
- HM,
- SA,
- `refined_intervals.csv`.

2. `05_confirmatory/02_wave2`:
- LHS wave2 внутри confirmatory refined-intervals,
- тот же iterative sigma-tuning (те же `max_iters`, `step`, `nroy_min/max`),
- пересчет HM и SA.

3. Сбор метрик стабильности и запись в `04_master/confirmatory_summary.json`.

Идея: методологически тот же контур, но на другом дизайне.

---

## 11.4 Как именно считается стабильность

### 11.4.1 Стабильность по NROY

Для main и confirmatory wave2 берется `history_matching.csv`, после чего:

\[
NROY\_pct = 100 \cdot \frac{\#\{nroy=True\}}{N}
\]

Далее:

\[
\Delta_{NROY}^{pp}=NROY_{confirm}-NROY_{main}
\]

и проверяется модуль отклонения:

\[
|\Delta_{NROY}^{pp}| \le tol
\]

где `tol = confirmatory_nroy_tol_pp` (по протоколу и run-config обычно `5.0` п.п.).

### 11.4.2 Стабильность по SA-top2

Из `sensitivity_uncertainty.csv` берутся компоненты с `rank_at_max_reduction <= 2`:

\[
T = \{component: rank\_at\_max\_reduction \le 2\}
\]

После этого сравниваются множества:

\[
sa\_top2\_stable = (T_{main} = T_{confirm})
\]

Важно: сравниваются именно **множества**, порядок не учитывается.

---

## 11.5 Критерий приемки confirmatory

Confirmatory-check считается пройденным только если одновременно:

1. `abs(nroy_delta_pp) <= confirmatory_nroy_tol_pp`
2. `sa_top2_stable == True`

В `quality_gates.csv` это отдельная строка:

`Confirmatory stability (|NROY delta| <= 5.0pp and SA top2 stable)`

Если этот check падает, итоговый scientific verdict (`PASS/FAIL`) тоже падает, даже если остальные блоки успешны.

---

## 11.6 Какие артефакты формируются

1. `05_confirmatory/01_wave1/*`
- полный набор wave1-артефактов confirmatory ветки (`lhs_runs.csv`, `history_matching.csv`, `sensitivity_uncertainty.csv`, `targets_calibrated.json`, и т.д.).

2. `05_confirmatory/02_wave2/*`
- полный набор wave2-артефактов,
- включая `sigma_tuning_iterations.csv` и итерационные `targets_tuned_iter*.json`/`targets_calibrated_iter*.json`.

3. `04_master/confirmatory_summary.json`
- агрегированный summary стабильности:
- `main_nroy_pct`
- `confirm_nroy_pct`
- `nroy_delta_pp`
- `main_sa_top2`
- `confirm_sa_top2`
- `sa_top2_stable`

4. `04_master/research_core_tables/quality_gates.csv`
- финальный PASS/FAIL по confirmatory-строке.

5. `04_master/research_core_report.md`
- секция `Confirmatory Stability` с итоговыми числами.

---

## 11.7 Как интерпретировать PASS и FAIL на практике

Пример из full-контура (`output_research_core_v2_full`):
- `main_nroy_pct = 54.91`
- `confirm_nroy_pct = 59.68`
- `nroy_delta_pp = 4.77`
- `main_sa_top2 = [MD, OU]`
- `confirm_sa_top2 = [MD, OU]`

Итог: confirmatory gate = PASS (дельта в пределах 5 п.п. и SA-top2 стабильна).

Пример из dry-контура (`output_research_core_v2_dry`):
- `main_nroy_pct = 93.75`
- `confirm_nroy_pct = 82.35`
- `nroy_delta_pp = -11.40`
- `sa_top2_stable = true`

Итог: confirmatory gate = FAIL (SA стабильна, но дельта NROY слишком велика по модулю).

---

## 11.8 Частые ошибки интерпретации

1. Ошибка: "NROY должны совпасть точно".
- Неверно: проверяется допуск по модулю (`<= 5` п.п.), а не точное равенство.

2. Ошибка: "Важно совпадение порядка top-2".
- Неверно: в коде сравниваются множества компонент, порядок не важен.

3. Ошибка: "top-2 всегда ровно две компоненты".
- Неверно: из-за dense-рангов и tie может получиться больше двух компонент в множестве (например, несколько компонент с одинаковым рангом 2).

4. Ошибка: "Confirmatory заменяет representative long-run".
- Неверно: это другой слой проверки; representative отвечает за long-run viability траекторий, confirmatory — за воспроизводимость статистического вывода HM/SA.

---

## 11.9 Что делать, если confirmatory не проходит

1. Если проваливается только `|nroy_delta_pp|`:
- пересмотреть строгость sigma-tuning,
- проверить, не переужесточен/переослаблен wave2 таргет-контур,
- повторить main+confirmatory симметрично.

2. Если проваливается `sa_top2_stable`:
- проверить `sensitivity_uncertainty.csv` в обеих ветках,
- оценить tie-сценарии рангов,
- уточнить неопределенности (OU/EV/MD/CU), чтобы ранжирование стало устойчивее.

3. Если падают оба условия:
- рассматривать это как признак неустойчивого вывода,
- не выносить итог в научный PASS без повторной стабилизации.

---

## 11.10 Ключевая мысль раздела 11

Confirmatory-run делает результат не просто "подогнанным в одном запуске", а воспроизводимо устойчивым.

Он отвечает на главный вопрос научной надежности:

> "Останется ли вывод тем же, если повторить весь HM/SA-контур на независимом дизайне?"

Если ответ "да" по формальным критериям, итоговый PASS становится методологически защищенным.
