# 6) Шаблон презентации (10–12 слайдов)

## Слайд 1 — Контекст и цель

- Что моделируем: ABM экономики (HH–Firms–Bank)
- Цель: анализ чувствительности, сокращение неопределённости параметров, стабильный отчётный набор сценариев

## Слайд 2 — Input → ABM → Output

- Вход: `theta` (13 параметров) + `seed`
- ABM: рынки + кредит + дефолты + банк (bail‑in/bailout)
- Выход: временные ряды агрегатов + метрики по окну

## Слайд 3 — Параметры (что меняем)

- Таблица bounds (baseline vs seminar) из `reports/2026-01-25_seminar/03_results.md` (раздел 3.1)

## Слайд 4 — Дизайн эксперимента (LHS + seeds)

- wave1: LHS (N точек) × seeds (0/1/2) → датасет
- wave2: повтор в refined bounds

Файлы: `output_seminar/datasets/lhs_runs*.csv`

## Слайд 5 — Эмулятор (surrogate model)

- GPR как основной эмулятор (даёт неопределённость)
- RF как контроль
- Метрика качества: CV R²

Файл: `output_seminar/results/emulator_scores.csv`

## Слайд 6 — History matching / NROY

- improbability `I`, агрегирование в `I_max`
- NROY: `I_max < threshold`

Файлы: `output_seminar/results/history_matching.csv`, `.../refined_intervals.csv`

## Слайд 7 — Сравнение результатов (baseline vs seminar)

- Таблица NROY/I_max и качества эмулятора из `reports/2026-01-25_seminar/03_results.md` (разделы 3.3–3.4)

## Слайд 8 — Стабильность отчётного набора (главное для демо)

- Показать, что в baseline был коллапс `rep_id=4`, а в seminar — нет
- Таблицы из `reports/2026-01-25_seminar/03_results.md` (раздел 3.6)

## Слайд 9 — Графики временных рядов (rep_run)

Рекомендованные графики по 6 rep_id (по 3 seed можно среднее/полоса):

- Employment, Output, Consumption
- AvgPrice
- HH_Deposit, HH_Debt
- Bank_Equity, BankResolutionHaircut, BankResolved/BailedOut индикаторы

Файлы: `output_seminar/results/report_set_wave2/rep_run_XX_seedY.csv`

## Слайд 10 — ML‑политика фирм (если нужно)

- что учим: `realized_markup`
- откуда данные: хвосты репрезентативных траекторий

Файлы: `output_seminar/ml_datasets/firm_policy.csv`, `output_seminar/models/firm_policy.pkl`

## Слайд 11 — Ограничения и интерпретация

- “плохие режимы” существуют (коллапсы), но в seminar‑пайплайне они отделены через `bad_run`
- `HH_Deposit_mean==0` остаётся частым даже среди good runs — кандидат на будущую донастройку

## Слайд 12 — Следующие шаги

- Optuna: оптимизация `theta` внутри NROY/безопасных bounds
- расширение таргетов и более строгие метрики устойчивости

