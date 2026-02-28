# Раздел 1. Ядро ABM-модели: агенты, состояния, механика и метрики

Этот раздел фиксирует ядро реализованной модели (`src/synpop/model.py`, `src/synpop/agents.py`, `src/synpop/bank.py`): кто именно является агентами, какие у них состояния, как идет шаг симуляции и какие макрометрики снимаются.

## 1.1 Состав агентов и экономические роли

В модели три типа агентов.

1. Домохозяйства (`Household`)
- состояние занятости (`employed`, `employer_id`)
- депозит и долг (`deposit`, `debt`)
- поведенческие параметры (`alpha`, `reserve_wage`, `skill`)
- решение о потреблении, погашении долга, запросе кредита
- дефолт при перегрузе долгом относительно дохода

2. Фирмы (`Firm`)
- производственные параметры (`productivity`)
- финансы и баланс (`cash`, `debt`, `inventory`)
- труд (`workers`, найм/увольнение)
- цена и наценка (`price`, `markup`, `base_markup`)
- кредит на ФОТ и ликвидность, выпуск, продажи, дефолт и re-entry после простоя

3. Банк (`Bank`)
- баланс: кредиты HH/firm, депозиты HH/firm, капитал, резервы
- кредитное предложение с лимитом по капиталу и prudential-rationing
- начисление процентов по кредитам/депозитам
- поглощение кредитных потерь
- разрешение неплатежеспособности через bail-in и, при необходимости, bailout

Ключевой принцип: макродинамика не задается напрямую, а возникает из микро-решений этих агентов в замкнутом stock-flow контуре.

## 1.2 Что задается при инициализации

Параметры ядра делятся на блоки:

- размер экономики: `n_households`, `n_firms`
- технологии и цены: `wage`, `productivity`, `base_price`, `price_elasticity`, `quality_weight`
- финансы: `loan_rate`, `deposit_rate`, `bank_credit_multiplier`
- рынок труда и ожидания: `adaptation_rate`, `demand_expectation_memory`, `max_hire_per_step`, `max_fire_per_step`
- блок ценообразования/запасов: `price_adjust_speed`, `price_stickiness`, `max_price_step`, `inventory_target_days`
- кредитный underwriting: `hh_max_dsr`, `firm_max_dsr`
- дефолты и потери: `hh_default_trigger_multiplier`, `firm_default_trigger_multiplier`, `hh_loss_given_default`, `firm_loss_given_default`
- потребление и сглаживание: `consumption_memory`, `precautionary_saving`, `consumption_floor_prop`, `consumption_ceiling_prop`, `unemployed_consumption_penalty`, `unemployment_transfer`

Начальные состояния:

- HH: `deposit_i(0)` из распределения, `debt_i(0)=0`, `employed` по стартовой занятости
- Firm: `cash_j(0)` из распределения, `debt_j(0)=0`, `inventory_j(0)=0`, начальная цена `base_price*(1+markup)`
- Bank: `equity_0` и `reserves_0` (в коде инициализируются от стартового кэша фирм), нулевые кредиты/депозиты

## 1.3 Стартовый спрос (инициализация ожиданий фирм)

В `initial_demand_share()` рассчитывается стартовый ориентир спроса на фирму:

\[
\text{ExpectedWageBill}=(N_H\cdot e_0)\cdot w
\]
\[
\text{ExpectedConsumption}=\text{ExpectedWageBill}\cdot \bar{\alpha}
\]
\[
\text{TotalUnits}=\frac{\text{ExpectedConsumption}}{p_0}
\]
\[
\text{InitialDemandPerFirm}=\frac{\text{TotalUnits}}{N_F}
\]

Этот показатель инициализирует `last_demand` и `expected_sales_ewma` фирм, после чего спрос уже полностью эндогенный.

## 1.4 Как моделируется один шаг \(t \rightarrow t+1\)

Фиксированный порядок блоков в `EconomyModel.step()`:

1. `begin_step()` банка и агентов
2. `bank.accrue_interest(...)`  
кредитные проценты начисляются, обслуживание ограничено ликвидностью заемщика; депозитный процент выплачивается только в пределах реально собранной процентной маржи
3. `_labor_market()`  
фирмы пересчитывают таргет занятости из ожидаемых продаж и запасов, затем нанимают/увольняют
4. `_production_and_wages()`  
фирма обновляет цену, при необходимости берет кредит на ФОТ, платит зарплаты (включая частичную выплату), производит выпуск и пополняет запасы
5. `_consumption_market()`  
HH решают потребление (погашение долга -> целевое потребление -> кредит при нехватке), затем совокупный спрос распределяется по фирмам через softmax по цене/качеству
6. `_handle_defaults()`  
проверка дефолтов HH и firm, списание потерь банка (LGD), reset состояний дефолтнувших агентов
7. `bank.update_balance_sheet(...)` и `_check_balance()`  
пересчет банковского баланса, при отрицательном капитале bail-in/bailout, затем контроль `Assets=Liabilities`
8. `datacollector.collect(self)`  
запись макроагрегатов шага

## 1.5 Механики устойчивости ядра (вшитые стабилизаторы)

- кредитный лимит банка:  
\[
CreditCap=\mu\cdot \max(Equity,0),\quad AvailableCredit=\max(0, CreditCap-Loans)
\]
- prudential rationing кредита при высокой утилизации капа
- DSR-based underwriting для HH и firm (не только cap банка, но и capacity заемщика)
- ограниченный шаг изменения цены + ценовая инерция (`price_stickiness`)
- сглаживание спроса и ожиданий (`demand_smoothing`, `expected_sales_ewma`)
- re-entry фирм после дефолта с лагом (`firm_reentry_lag`)
- проверка балансовой консистентности с `balance_tolerance`

## 1.6 Полный список метрик, снимаемых из ядра ABM

`DataCollector` записывает:

- рынок труда: `Employment`, `UnemploymentRate`
- реальный сектор: `Output`, `Production`, `Consumption`, `WageBill`
- трансферты: `Transfers`, `UnemploymentTransfers`
- балансы HH/firm: `HH_Deposit`, `HH_Debt`, `Firm_Cash`, `Firm_Debt`, `Inventories`
- цены и запасы: `AvgPrice`, `AvgMarkup`, `PriceDispersion`, `InventoryGap`, `InventoryTurnover`, `SalesForecastError`
- кредитный канал: `CreditRequests`, `CreditRejections`, `CreditRejectionRate`
- динамика фирм после дефолтов: `FirmDowntimeShare`, `ReentryCount`
- банк: `Bank_Equity`, `Bank_Loans`, `Bank_Deposits`, `BankFailed`, `BankResolved`, `BankResolutionAmount`, `BankResolutionHaircut`, `BankBailedOut`, `BankBailoutAmount`
- дефолты и консистентность: `Defaults`, `DefaultsHH`, `DefaultsFirm`, `BalanceOK`

Именно этот набор далее агрегируется в `*_mean` и используется в калибровочном pipeline (LHS -> emulator -> history matching -> wave2).

## 1.7 Что важно по итогу раздела

1. Ядро ABM уже реализует полный микро-макро контур: труд, производство, потребление, кредит, дефолты, банковское разрешение.
2. Все ключевые устойчивостные механизмы встроены на уровне правил агентов, а не пост-обработки.
3. Набор метрик покрывает и макродинамику, и структурные риски (кредитный канал, дефолты, баланс, банк), поэтому пригоден для формальной калибровки и quality-gates.
