# 5) Каталог параметров (полный)

## 5.1 Параметры `EconomyModel` (все аргументы конструктора)

Источник: сигнатура `EconomyModel.__init__` в `src/synpop/model.py`.

Формат: `имя = default`.

### Размер и базовые настройки

- `n_households = 100`
- `n_firms = 10`
- `seed = None`
- `enable_credit = True`

### Поведение/технологии (часть варьируется в калибровке)

- `wage = 1.0` *(варьируется через theta)*
- `productivity = 1.0` *(варьируется через theta)*
- `alpha_mean = 0.85` *(варьируется через theta)*
- `alpha_std = 0.05` *(варьируется через theta)*
- `adaptation_rate = 0.5` *(варьируется через theta)*
- `price_elasticity = 2.0` *(варьируется через theta)*
- `skill_wage_weight = 0.0` *(варьируется через theta)*
- `demand_smoothing = 0.1` *(варьируется через theta)*
- `quality_weight = 0.0`

### Инициализация

- `initial_savings_range = (0.0, 2.0)`
- `firm_initial_cash = 5.0`
- `initial_employment_rate = 0.8`

### Финансовые параметры (часть варьируется в калибровке)

- `loan_rate = 0.02` *(варьируется через theta)*
- `deposit_rate = 0.005` *(варьируется через theta)*
- `bank_credit_multiplier = 6.0` *(варьируется через theta)*
- `hh_debt_cap_multiplier = 4.0` *(варьируется через theta)*
- `firm_debt_cap_multiplier = 2.0` *(варьируется через theta)*
- `repayment_fraction = 0.1`

### Дефолты и потери (fixed defaults)

- `hh_default_trigger_multiplier = 1.5`
- `firm_default_trigger_multiplier = 1.5`
- `hh_loss_given_default = 0.6`
- `firm_loss_given_default = 0.6`

### Цены и численные настройки

- `base_price = 1.0`
- `balance_tolerance = 1e-6`
- `log_balance_warnings = True`
- `demand_floor = None`

### Расширенные конфиги (опционально)

- `household_config = None`
- `firm_config = None`
- `controls = None`

## 5.2 Варьируемые параметры (theta) и их bounds

См. `reports/2026-01-25_seminar/03_results.md` (раздел 3.1).

## 5.3 Таргеты для history matching

Источник: `scripts/targets_report.json`.

Метрики:

- `Employment_mean`
- `Output_mean`
- `Consumption_mean`
- `AvgPrice_mean`
- `DefaultsHH_rate`
- `BankFailed_share`
- `BankResolved_share`
- `BankResolutionHaircut_mean`
- `BankBailedOut_share`
- `BankBailoutAmount_mean`
- `Transfers_mean`
- `HH_Deposit_mean`
- `Bank_Equity_mean`

Каждый таргет задаётся тройкой: `target`, `sigma_obs`, `sigma_model`.

