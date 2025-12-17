# SynPop / ABM макроэкономики (Mesa)

Проект: агент-ориентированная макроэкономическая модель на `Mesa` (домохозяйства + фирмы + банк), сценарные прогоны и калибровка через LHS → эмулятор (GPR/RF) → history matching (NROY) → сужение интервалов (wave2) → подбор репрезентативных прогонов для отчёта.

## Структура репозитория

- `src/synpop/` — ядро модели:
  - `model.py` — `EconomyModel` (Mesa Model), сбор метрик через `DataCollector`
  - `agents.py`, `bank.py` — агенты и банковский сектор
  - `scenarios.py` — сценарии, `summarize_run`, генерация таблиц/графиков
- `src/synpop_model.py` — простой entrypoint/шима для запуска
- `scripts/` — утилиты пайплайна:
  - `run_lhs.py` — генерация LHS-дизайна и прогон батча параметров → датасет
  - `train_emulator.py` — обучение эмулятора (GPR + RF baseline) + history matching + refined intervals
  - `make_report_set.py` — выбор «репрезентативных» точек и длинные прогоны (таймсерии) для отчёта
  - `make_calibration_summary.py` — сборка краткого `output/results/calibration_summary.md`
  - `make_scenario_gif.py` — сборка GIF по сценариям (если используешь)
- `output/` — артефакты запусков (датасеты, результаты, графики, GIF)
- `theory/` — теоретические материалы (PDF/DOCX)

## Требования

- Python: 3.10+
- Библиотеки (минимально по коду репо): `mesa`, `numpy`, `pandas`, `scikit-learn`
- Для графиков/картинок: `matplotlib`, `pillow`
- Для Parquet (если выберешь `--format parquet`): `pyarrow`

Установка (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install mesa numpy pandas scikit-learn matplotlib pillow pyarrow
```

Важно: чтобы импорты `synpop` работали без упаковки проекта, выставляй `PYTHONPATH` на `src/`.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## Быстрый старт (демо и сценарии)

Демо-прогон:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m synpop_model
```

Сценарный набор (сохранить CSV + summary + график в `output/`):

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -c "from synpop import run_scenarios; run_scenarios(steps=120, window=40, save=True, plot=True, output_dir='output')"
```

Что появится в `output/`:
- `scenario_*.csv` — таймсерии по каждому сценарию
- `scenario_summary.csv` — агрегаты по хвосту окна `window`
- `scenario_timeseries.csv` — все сценарии одной таблицей
- `scenario_plot.png` — сводный график (если `plot=True`)
- `scenario.gif` — если запускал `scripts/make_scenario_gif.py`

## Пайплайн калибровки (LHS → эмулятор → history matching → wave2)

Ниже команды соответствуют тому, что уже лежит у тебя в `output/datasets` и `output/results`.

### 1) Wave1: LHS датасет

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python .\scripts\run_lhs.py --n 200 --seeds 0,1,2 --steps 150 --window 30 --format csv --output output/datasets/lhs_runs.csv
```

Выход: `output/datasets/lhs_runs.csv` (по строке на (theta, seed) + агрегированные метрики `*_mean/_rate/_share`, и флаги `bad_run/error`).

### 2) Wave1: обучение эмулятора + history matching

`targets` для HM лежат в `scripts/targets_report.json`.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python .\scripts\train_emulator.py --data output/datasets/lhs_runs.csv --targets scripts/targets_report.json --outdir output/results
```

Выход в `output/results/`:
- `emulator_scores.csv` — качество GPR/RF по метрикам (R2/RMSE + CV)
- `history_matching.csv` — таблица с `*_pred`, `*_sigma_emul`, `*_I`, итоговые `I_max` и `nroy`
- `refined_intervals.csv` — суженные интервалы параметров по NROY (min/max и p05/p95 + `shrink_pct`)

### 3) Wave2: LHS внутри refined-интервалов

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python .\scripts\run_lhs.py --n 200 --seeds 0,1,2 --steps 150 --window 30 --bounds-csv output/results/refined_intervals.csv --bounds-kind p05p95 --format csv --output output/datasets/lhs_runs_wave2.csv
```

### 4) Wave2: эмулятор + HM (в отдельную папку)

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python .\scripts\train_emulator.py --data output/datasets/lhs_runs_wave2.csv --targets scripts/targets_report.json --outdir output/results/wave2
```

### 5) Репрезентативные прогоны для отчёта

Идея: из датасета выбрать несколько «читаемых» точек (с фильтрами по Employment/Output/Defaults/банк‑событиям), затем для каждой точки сделать длинные прогоны и сохранить таймсерии.

Wave1:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python .\scripts\make_report_set.py --data output/datasets/lhs_runs.csv --history-matching output/results/history_matching.csv --outdir output/results/report_set --n 6 --steps 300 --window 50 --seeds 0,1,2
```

Wave2:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python .\scripts\make_report_set.py --data output/datasets/lhs_runs_wave2.csv --history-matching output/results/wave2/history_matching.csv --outdir output/results/report_set_wave2 --n 6 --steps 300 --window 50 --seeds 0,1,2
```

Формат выхода (`output/results/report_set*`):
- `representative_points.csv` — выбранные точки + их агрегаты/score
- `representative_summary.csv` — сводка по (точка, seed), включает `bad_run/error`, `nroy`, `I_max`, `score`
- `rep_run_XX_seedY.csv` — таймсерия по шагам (колонки: `Employment`, `Output`, `Consumption`, `HH_Deposit`, `BankFailed/Resolved/BailedOut`, `Defaults*`, `BalanceOK`, …)

### 6) Короткий итог по калибровке

```powershell
python .\scripts\make_calibration_summary.py --out output/results/calibration_summary.md
```

## Что лежит в `output/results/` (смысл файлов)

- `output/results/emulator_scores.csv` — качество эмулятора по каждой метрике: какие метрики обучались/пропущены, и качество `GPR`/`RF` (holdout + CV).
- `output/results/history_matching.csv` — для каждой LHS-точки: прогноз/дисперсия эмулятора, improbability `I` по каждой целевой метрике, затем `I_max` и флаг `nroy`.
- `output/results/refined_intervals.csv` — сужение интервалов параметров по NROY: `nroy_min/max`, `nroy_p05/p95`, `shrink_pct`.
- `output/results/wave2/…` — то же самое для второй волны (после сужения интервалов).
- `output/results/report_set*` и `output/results/report_final/` — репрезентативные точки и длинные таймсерии (удобно для графиков/таблиц в отчёте).
- `output/results/calibration_summary.md` — человекочитаемая выжимка «wave1 vs wave2».

## Пример результатов текущего прогона

См. `output/results/calibration_summary.md`.

## Частые проблемы

- `ModuleNotFoundError: synpop` → выставь `PYTHONPATH`:
  - PowerShell: `$env:PYTHONPATH = (Resolve-Path .\src).Path`
- Датасет Parquet не пишется/читается → установи `pyarrow` или используй `--format csv`
- В `run_lhs.py` много «плохих» прогонов (`bad_run=True`) → сузь bounds (или используй `refined_intervals.csv`), проверь `scripts/run_operator.py:PARAM_BOUNDS`
