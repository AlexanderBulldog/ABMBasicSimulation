# Раздел 12. Quality Gates и финальный научный вердикт

Этот раздел фиксирует, как из ABM/HM/SA/representative/confirmatory собирается формальный `PASS/FAIL`.

## 12.1 Почему quality gates обязательны

Quality gates делают вывод:
- воспроизводимым,
- проверяемым,
- независимым от субъективной интерпретации графиков.

Если хотя бы один blocking-gate не пройден, общий verdict = `FAIL`.

---

## 12.2 Два контура: active и legacy

В актуальном протоколе используются два контура:

1. `stability` (active-v3, blocking)
- участвует в финальном scientific verdict.

2. `legacy/reference` (не blocking)
- нужен как эталон сравнения, но не блокирует итоговый PASS/FAIL.

Пример: одновременно присутствуют строки
- `NROY wave2 in [25,65]%` (blocking)
- `NROY wave2 in [25,60]%` (reference)

---

## 12.3 Актуальные blocking-гейты (v3)

### 12.3.1 HM и эмулятор

- `NROY wave2 in [25,65]%`
- `I_max median wave2 < 3.0`
- `I_max p95 wave2 < 4.5`
- `Emulator CV-R2 median >= 0.60`
- `Emulator CV-R2 share>=0.30 >= 0.65`

### 12.3.2 ABM / representative

- `bad_run_pct_w2 <= 5%`
- `Representative Employment_mean >= 45`
- `Representative Output_mean >= 50`
- `Representative Consumption_mean >= 50`
- `Representative BankFailed_share == 0`
- `Representative BalanceOK_share == 1`

### 12.3.3 SA и confirmatory

- `SA components EV/OU/MD/CU present`
- `SA has >=16 rows`
- `SA ranks present`
- `Confirmatory stability (|NROY delta| <= 5.0pp and SA top2 stable)`

### 12.3.4 Structural (blocking)

- `Structural: PriceDispersion median in [0.03,0.30]`
- `Structural: |InventoryGap| p95 <= 1.5`
- `Structural: CreditRejectionRate mean <= 0.80`

---

## 12.4 Формула итогового verdict

Пусть `B` — множество blocking-строк из `quality_gates.csv`.

\[
overall\_pass = \bigwedge_{g \in B} pass_g
\]

Legacy/reference строки исключены из этой конъюнкции.

---

## 12.5 Актуальный результат по full-run v6

Источник: `output_research_core_v6_full_20260214_182812/04_master/research_core_tables/quality_gates.csv`.

Blocking FAIL:
- `NROY wave2 in [25,65]%` -> `68.40%` (выше верхней границы)
- `Emulator CV-R2 median >= 0.60` -> `0.5570`
- `Confirmatory stability` -> `nroy_delta_pp=15.31`

Blocking PASS:
- `I_max median=2.4292`, `I_max p95=2.9273`
- representative floors и банковая устойчивость
- SA полнота
- structural checks (PriceDispersion/InventoryGap/CreditRejectionRate mean)

Итог по v6: `overall_gate_pass_v3 = FAIL`.

---

## 12.6 Актуальный результат по evidence v3

Источник: `output_evidence_campaign_v3_20260215/campaign_evidence_summary.json`.

- Stage A: 4/4 кандидата прошли core-pass (после retry где нужно)
- Winner: `A4`
- Stage B (`R1/R2/R3`): `pass_runs_count = 0`
- Acceptance rule: `core_pass_at_least_2_of_3_and_consistency_ok`
- Campaign status: `FAIL`

Ключевые блокеры Stage B:
- confirmatory instability,
- для `R1/R2/R3` дополнительно провалы representative floors по занятости/выпуску/потреблению.

---

## 12.7 Что означает текущий FAIL

Текущий FAIL не означает, что ядро ABM нерабочее.

Он означает, что при текущих настройках и seed-репликациях не выполнены одновременно:
- достаточная дискриминация NROY,
- стабильность качества эмулятора,
- воспроизводимость confirmatory на уровне, требуемом blocking-протоколом.

---

## 12.8 Ключевая мысль раздела 12

Quality gates в текущем состоянии дают однозначный и аудируемый вывод:
- локальные успехи есть (часть волн и Stage A),
- но финальный scientific verdict на последних данных (`v6`, `evidence_v3`) остается `FAIL` из-за воспроизводимости и стабильности по blocking-критериям.
