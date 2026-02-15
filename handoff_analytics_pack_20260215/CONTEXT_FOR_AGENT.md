# Context Package for External Analytics Agent

## 1) Objective of this handoff
Analyze the current state of the ABM + Emulator + History Matching research pipeline and produce a concise, evidence-based assessment for reporting (`WorkPlan` / `WorkStatus` refresh).

Primary question:
- What is scientifically solid already, what is still weak, and what should be prioritized next to increase evidence strength?

## 2) Project-level alignment (global plan)
Global project items in scope:
1. Building and validation of probabilistic forecasting models for dynamics of epidemiological, economic, ecological, and social processes.
2. Resulting probabilistic models and validation outcomes.

Current work package maps to the economic branch:
- ABM macroeconomy simulation + surrogate modeling (GPR emulator) + Bayesian-style history matching with uncertainty decomposition and quality gates.

## 3) Individual plan alignment
### Done (implemented and run)
- Literature-informed pipeline in the spirit of Andrianakis HM tutorial/case-study.
- Problem formulation for heterogeneous agent economy.
- ABM run protocol with stochastic seeds and target metrics.
- LHS wave workflow, emulator training diagnostics, HM/NROY gating.
- Quality gates and scientific verdict logic (blocking vs reference contour).
- Multiple preflight/full/evidence campaigns with run artifacts.

### In progress
- Stabilizing replication-level evidence (2/3 criterion).
- Improving explainability and compact reporting for seminar/supervisor.
- Tightening consistency between preflight and full-run behavior.

### Next
- Stronger replication acceptance under fixed protocol.
- Final report-quality visualization and narrative with minimal ambiguity.
- Clear risk-to-action mapping for unresolved gates.

## 4) What these files contain
- `WorkPlan.pdf`, `WorkStatus.pdf`: original baseline documents to be reworked.
- `sources/quality_gates_baseline_v3.csv`: baseline gate table.
- `sources/quality_gates_current_sanityA1.csv`: current (sanity) gate table.
- `sources/research_core_report_sanityA1.md`: consolidated scientific verdict and wave diagnostics.
- `sources/waves_summary_sanityA1.csv`: wave-by-wave convergence metrics.
- `sources/campaign_evidence_summary.json`: Stage A/B/C evidence campaign result (includes winner, replications, verdict).
- `sources/evidence_*.log`: execution logs for campaign diagnostics.
- `sources/10_representative_long_runs_selection_ru.md`: representative long-run methodology.
- `sources/12_quality_gates_and_scientific_verdict_ru.md`: quality gate interpretation and verdict logic.
- `plots/*.png`: representative trajectory visuals for quick qualitative context.

## 5) Recommended analytics tasks for receiving agent
1. Build a strict baseline-vs-current table from gate CSVs (same units, same thresholds).
2. Validate consistency between `research_core_report_sanityA1.md`, `waves_summary_sanityA1.csv`, and `campaign_evidence_summary.json`.
3. Identify top-3 technical bottlenecks preventing stronger replication evidence.
4. Produce a compact plan for next run cycle: what to freeze, what to tune, what to monitor.
5. Draft report skeleton sections with source-traceable claims only.

## 6) Notes on interpretation
- Legacy/reference contour is informative, but blocking scientific verdict should prioritize v3 blocking gates.
- Preflight success does not automatically imply multi-replication success in full campaign.
- Treat replication acceptance criterion (2/3 + consistency) as high-priority evidence threshold.

## 7) Package timestamp
Prepared: 2026-02-15 (local workspace snapshot)
