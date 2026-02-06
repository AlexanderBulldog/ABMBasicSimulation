# Одностраничный исследовательский статус (на 6 февраля 2026)

## Контекст и цель
Проект развивает ABM-модель макроэкономики (`домохозяйства + фирмы + банк`) и калибровочный контур:
`LHS -> эмулятор (GPR/RF) -> history matching (NROY) -> wave2 -> representative runs -> ML-policy`.

Цель текущего этапа: получить стабильный и интерпретируемый набор результатов для семинара/статьи и проверить воспроизводимость ключевых компонентов.

## Ключевые достижения
1. Рабочий вычислительный контур калибровки.
   Основные артефакты пересобраны и актуализированы:
   - `output/results/calibration_summary_fresh_2026-02-06.md`
   - `output_seminar/results/calibration_summary_fresh_2026-02-06.md`

2. Семинарная ветка показывает более пригодное поведение для исследовательской интерпретации.
   По `output_seminar/results/calibration_summary_fresh_2026-02-06.md`:
   - Emulator quality (wave1): `GPR CV R2 mean=0.607`, `median=0.671`
   - History matching: `NROY wave1=53.3%`, `wave2=48.5%`
   - После фильтрации `bad_run`:
     `Employment_mean==0 / Output_mean==0 / Consumption_mean==0 = 0.0%` (good runs)

3. Репрезентативный отчётный набор стабилен на длинном горизонте.
   По `output_seminar/results/report_set_wave2/representative_summary.csv` (18 прогонов: 6 точек x 3 seed):
   - `Employment_mean`: min `56`, avg `68.67`
   - `Output_mean`: min `78.29`, avg `88.90`
   - `Consumption_mean`: min `72.65`, avg `87.79`
   - `BankBailedOut_share`: min/max `0/0`
   - `BankFailed_share`: min/max `0/0`
   - `BalanceOK_share`: min/max `1/1`

4. Компоненты исполняются end-to-end (smoke validation).
   На малом прогоне отработали:
   - `scripts/run_lhs.py`
   - `scripts/train_emulator.py`
   - `scripts/make_report_set.py`
   - `scripts/plot_report_runs.py`
   - `scripts/build_firm_policy_dataset.py`
   - `scripts/train_firm_policy.py`
   Результаты smoke-проверки: `output/smoke/*` (включая `output/smoke/models/firm_policy.pkl`).

## На чём остановилась разработка
1. Исследовательская логика и артефакты уже в рабочем состоянии, но инженерная инфраструктура не доведена до production-уровня:
   - нет формализованного файла зависимостей проекта (`pyproject.toml`/`requirements.txt`);
   - локальный `.venv` привязан к отсутствующему интерпретатору (`pyenv`), запуск идёт через `py -3.13`;
   - нет автоматических тестов/CI.

2. Научно-прикладной стоп-поинт:
   - базовая ветка (`output/`) содержит много вырожденных режимов;
   - семинарная ветка (`output_seminar/`) уже даёт стабильный report set и используется как основа для включения в исследование;
   - модуль ML-policy готов как дополнительный (exploratory) блок, но пока не центральный источник доказательств.

## Вывод для включения в исследование
Сейчас обоснованно включать в основную часть:
- ABM-ядро и механизм банка/дефолтов,
- LHS + эмулятор + history matching + wave2,
- representative trajectories из `output_seminar/results/report_set_wave2`.

Включать с оговоркой:
- ML-policy фирм как расширение и демонстрацию направления развития, а не как главный эмпирический результат.
