# 1) Спецификация модели (ABM)

## 1.1 Ядро модели

Модель реализована как Mesa Model:

- `src/synpop/model.py` — `EconomyModel` (основной цикл симуляции + сбор метрик)
- `src/synpop/agents.py` — `Household`, `Firm` (правила поведения)
- `src/synpop/bank.py` — `Bank` (баланс, кредит, bail‑in/bailout)
- `src/synpop/builder.py` — генерация синтетической популяции

## 1.2 Шаг симуляции (макро‑петля)

Внутри `EconomyModel.step()` (см. `src/synpop/model.py`) выполняются ключевые блоки:

1) interest accrual (проценты по кредиту/депозиту)  
2) рынок труда (найм/увольнение)  
3) производство и выплаты заработной платы  
4) рынок потребления (спрос домохозяйств → продажи фирм)  
5) дефолты домохозяйств и фирм  
6) обновление баланса банка + проверка баланса

## 1.3 Поведение домохозяйств (Household)

См. `src/synpop/agents.py`:

- Доход: домохозяйство получает зарплату, накапливает депозит.
- Потребление: выбирает желаемое потребление как `alpha * (deposit + ожидание_дохода)` (см. `decide_consumption`).
- Если депозита недостаточно — может занять у банка, но с лимитом по долгу:
  - `credit_cap = hh_debt_cap_multiplier * last_income`
- Дефолт домохозяйства: если долг превышает порог (с мультипликатором триггера), банк несёт loss‑given‑default.

## 1.4 Поведение фирм (Firm)

См. `src/synpop/agents.py`:

- Ценообразование:
  - базово фирма использует markup‑правило с сигналом по запасам (inventory vs last_demand),
  - либо (опционально) использует ML‑политику `firm_policy` (предсказывает markup по состоянию фирмы).
  - markup ограничен `clamp(markup, 0.0, 0.5)`.
  - цена снизу ограничена: `price = max(0.1, unit_cost * (1 + markup))`.
- Производство:
  - фирма нанимает/увольняет по `adaptation_rate` и целевому числу работников,
  - может занимать у банка для выплаты зарплат/овердрафта (`bank.grant_loan`),
  - выпуск: `output = productivity * effective_labor`, поступает на склад (inventory).
- Дефолт фирмы:
  - сравнение долга с порогом от выручки (оценка из `last_revenue`/`last_demand*price`).

## 1.5 Банк (Bank)

См. `src/synpop/bank.py`:

- Доступный кредит:
  - `available_credit = credit_multiplier * equity - (loans_firms + loans_hh)`
  - + prudential factor, уменьшающий выдачу по мере исчерпания капитала.
- В случае отрицательного капитала:
  - сначала применяется **bail‑in** (haircut депозитов фирм/домохозяйств) до восстановления буфера капитала,
  - если депозитов нет/недостаточно — возможен **bailout** (внешняя докапитализация).
- Метрики банка фиксируются по шагам (`BankFailed`, `BankResolved`, `BankBailedOut`, haircut/amount).

## 1.6 Параметры модели

### 1.6.1 Варьируемые параметры (theta)

Это параметры, которые меняются в LHS/калибровке и являются “инпутом” оптимизации:

Задаются в `scripts/run_operator.py:21` (`PARAM_BOUNDS`).

- `alpha_mean`, `alpha_std`
- `wage`, `productivity`
- `loan_rate`, `deposit_rate`
- `bank_credit_multiplier`
- `hh_debt_cap_multiplier`, `firm_debt_cap_multiplier`
- `adaptation_rate`
- `price_elasticity`
- `skill_wage_weight`
- `demand_smoothing`

### 1.6.2 Остальные параметры (фиксированные по умолчанию)

Список с дефолтами — по сигнатуре `EconomyModel.__init__` (см. `src/synpop/model.py`).  
Ключевые примеры:

- размеры экономики: `n_households=100`, `n_firms=10`
- `initial_employment_rate=0.8`
- дефолтные триггеры/потери: `hh_default_trigger_multiplier`, `hh_loss_given_default`, и аналоги для фирм
- `base_price=1.0`
- `balance_tolerance=1e-6`

## 1.7 Метрики выхода (что измеряем)

Сводные метрики формируются из временных рядов как средние по последнему окну `window`:

- базовый список `*_mean`: `src/synpop/scenarios.py:summarize_run`
- дополнительные нормировки/доли: `scripts/run_operator.py:run_model`:
  - `DefaultsHH_rate`, `DefaultsFirm_rate`
  - `BalanceOK_share`
  - `BankFailed_share`, `BankResolved_share`, `BankBailedOut_share`

