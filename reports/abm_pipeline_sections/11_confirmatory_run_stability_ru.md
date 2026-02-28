# Раздел 11. Confirmatory-run: независимая проверка устойчивости вывода

Этот раздел описывает, как после основного цикла проверяется, что вывод не является артефактом одного LHS-дизайна.

## 11.1 Зачем нужен confirmatory-run

Даже если main-run проходит HM и structural checks, это не гарантирует воспроизводимость.

Confirmatory-run отвечает на вопрос:

> "Сохранится ли вывод при независимом повторе финальной волны с теми же правилами HM/SA?"

То есть это проверка устойчивости статистического вывода, а не дополнительная подгонка.

---

## 11.2 Что считается независимым в текущем ядре

Независимость задается сдвигом master-seed через `--confirm-seed-offset-base`.

Для кампании `evidence_v3_20260215` использовалось:
- `confirm_seed_offset_base = 31`
- отдельный запуск final-wave внутри `05_confirmatory/wave_XX`

При этом фиксируются:
- ABM-ядро,
- bounds финальной волны,
- правила HM/SA,
- quality gates.

---

## 11.3 Текущая структура confirmatory в пайплайне

В актуальном `scripts/run_research_core.py` confirmatory запускается для **финальной волны**, а не как отдельные `01_wave1/02_wave2`.

Артефакты:
- `05_confirmatory/wave_XX/lhs_runs_confirm_waveXX.csv`
- `05_confirmatory/wave_XX/history_matching.csv`
- `05_confirmatory/wave_XX/sensitivity_uncertainty.csv`
- `04_master/confirmatory_summary.json`

Ключевая идея: повторить именно финальную фазу сравнения main vs confirm на одном и том же протоколе.

---

## 11.4 Как считается стабильность

Из `confirmatory_summary.json` берутся:
- `main_nroy_pct`
- `confirm_nroy_pct`
- `nroy_delta_pp = confirm_nroy_pct - main_nroy_pct`
- `main_sa_top2`, `confirm_sa_top2`
- `sa_top2_stable`

Критерий stable-confirm в blocking-гейтах:

\[
|nroy\_delta\_pp| \le 5.0
\]

и

\[
sa\_top2\_stable = \text{True}
\]

Это отражается строкой:

`Confirmatory stability (|NROY delta| <= 5.0pp and SA top2 stable)`

в `quality_gates.csv`.

---

## 11.5 Факты по последним прогонам

### 11.5.1 `output_research_core_v6_full_20260214_182812`

- `nroy_delta_pp = 15.31`
- `sa_top2_stable = True`
- blocking confirmatory gate: `FAIL`

Итог: основной full-run не прошел по воспроизводимости, даже при стабильном SA-top2.

### 11.5.2 Stage A winner в `output_evidence_campaign_v3_20260215`

Победитель `A4` после quick-confirm:
- `main_nroy_pct = 64.17`
- `confirm_nroy_pct = 62.50`
- `nroy_delta_pp = -1.67`
- `sa_top2_stable = True`

Итог: mini-confirm фильтр пройден (дельта в допуске).

### 11.5.3 Stage B (`R1/R2/R3`) в `evidence_v3_20260215`

- `R1`: `nroy_delta_pp = 16.57` -> FAIL
- `R2`: `nroy_delta_pp = 31.43` -> FAIL
- `R3`: `nroy_delta_pp = 64.00`, `sa_top2_stable=False` -> FAIL

Итог: confirmatory-нестабильность остается главным репликационным риском.

---

## 11.6 Практическая интерпретация

PASS confirmatory означает:
- переносимость вывода на независимую выборку,
- отсутствие критической чувствительности к конкретному LHS-случаю.

FAIL confirmatory означает:
- вывод пока не воспроизводится стабильно,
- запуск нельзя считать достаточно сильным доказательством для итогового scientific PASS.

---

## 11.7 Что делать при confirmatory FAIL

1. Сохранять симметрию main/confirm по `n`, steps, bounds, sigma-policy.
2. Отсеивать кандидатов Stage A по mini-confirm до репликаций Stage B.
3. Снижать межзапусковую волатильность NROY (через устойчивые refined-bounds и conservative sigma-step).
4. Фиксировать причину отказа по `reason_code=confirmatory_unstable` в gate-таблицах.

---

## 11.8 Ключевая мысль раздела 11

Confirmatory-run в текущем контуре — это формальный тест воспроизводимости финального вывода.

На момент последних данных (v6 + evidence_v3) именно confirmatory остается критическим блокером перехода от локально хороших прогонов к устойчивому научному PASS.
