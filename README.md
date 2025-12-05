# Агент-ориентированная модель (ABM) экономики домохозяйств и фирм на Mesa

Упрощенная макро-модель с населением домохозяйств и фирм, банком и рынками труда/потребления. Используется для быстрых сценариев: кредитное сжатие/расширение, шоки спроса/издержек, адаптация цен и занятости.

## Структура репозитория
- `src/` — исходный код модели (`synpop/*`, шима `synpop_model.py`).
- `scripts/` — вспомогательные скрипты (например, `make_scenario_gif.py`).
- `output/` — сохраненные сценарии и графики.
- `theory/` — теоретические материалы и презентации (PDF/DOCX).

## Установка
Требуется Python 3.10+.
```powershell
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install mesa numpy pandas matplotlib pillow
```

## Быстрый старт
Запуск демо-сценария (печатает первые строки и финальные агрегаты):
```powershell
$env:PYTHONPATH="src"
python -m synpop_model
```

Минимальный пример из кода:
```python
from synpop import EconomyModel

m = EconomyModel(seed=1, n_households=80, n_firms=8, enable_credit=True)
m.run_model(steps=50)
df = m.results_dataframe()
print(df.tail())
```

## Сценарии и сохранение результатов
Готовый набор сценариев есть в `synpop/scenarios.py`.
```powershell
$env:PYTHONPATH="src"
python - <<'PY'
from synpop import run_scenarios

run_scenarios(steps=120, window=40, save=True, plot=True, output_dir="output")
PY
```
- При `save=True` будут сохранены `scenario_*.csv/png`, агрегаты `scenario_summary.csv` и `scenario_timeseries.csv` в `output/`.
- `plot=True` строит сравнение ключевых серий (либо сохраняет `scenario_plot.png` при `save=True`).

### GIF с шоком спроса/восстановлением
```powershell
python scripts/make_scenario_gif.py
```
Скрипт сам добавит `src/` в `PYTHONPATH` и сохранит GIF в `output/scenario.gif`.

## Ключевые элементы модели
- **Домохозяйства**: склонность к потреблению `alpha`, резервная зарплата, сберегательные/кредитные решения, возможен дефолт при превышении долгового лимита.
- **Фирмы**: производительность, денежные остатки, адаптация штата, ценообразование с наценкой и сигналом запасов, кредиты для выплаты зарплат, риск дефолта при высоком долге и отсутствии выручки.
- **Банк**: SFC-баланс — кредиты домохозяйствам/фирмам, депозиты, капитал, резервы; процентная маржа изменяет капитал; кредитное плечо через `bank_credit_multiplier`.
- **Рынок труда**: целевой штат от ожидаемого спроса (`adaptation_rate`), найм лучших по навыку, ограничения платежеспособностью при `enable_credit=False`.
- **Рынок товаров**: спрос формируется из доступных средств/кредита домохозяйств; распределение по фирмам через logit на цену/качество (`price_elasticity`, `quality_weight`), сглаживание ожиданий спроса (`demand_smoothing`, `demand_floor`).

## Основные параметры (фрагмент)
- `n_households`, `n_firms` — размеры популяций.
- `wage`, `productivity` — базовая зарплата/производительность; фактическая зарплата может множиться на навык (`skill_wage_weight`).
- `household_config` — генерация `alpha`, сбережений (uniform/lognormal), резервной зарплаты и навыка; `controls["households"]["total_savings"]` нормирует общие сбережения.
- `firm_config` — распределения производительности и наличности (normal/lognormal) и опциональные `market_shares`; `controls["firms"]["total_cash"]` нормирует общий кеш.
- `loan_rate`, `deposit_rate`, `bank_credit_multiplier`, `repayment_fraction` — кредитование и обслуживание долга.
- `adaptation_rate`, `initial_employment_rate` — скорость подстройки штата и стартовая занятость.
- `price_elasticity`, `quality_weight`, `base_price`, `demand_smoothing` — реакция спроса на цены/качество и сглаживание ожиданий.

## Форматы вывода
`model.results_dataframe()` возвращает DataFrame с сериями: `Employment`, `UnemploymentRate`, `Output`, `Production`, `Consumption`, `HH_Deposit`, `HH_Debt`, `Firm_Debt`, `Firm_Cash`, `Inventories`, `AvgPrice`, `Bank_Equity`, `Bank_Loans`, `Bank_Deposits`, `Defaults`, `BalanceOK`.

## Теория
Материалы в `theory/` (конспекты, презентации по ABM/макроэкономике) отсортированы отдельно от кода и артефактов моделирования.
