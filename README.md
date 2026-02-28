# ABM Core (SynPop)

Clean repository with only the core agent-based macro model and interaction mechanics:
- households
- firms
- bank balance-sheet and credit channel
- labor market, production, consumption, defaults, and accounting checks

## Structure

- `src/synpop/model.py` - `EconomyModel`
- `src/synpop/agents.py` - `Household`, `Firm`
- `src/synpop/bank.py` - bank and balance-sheet logic
- `src/synpop/builder.py` - synthetic population builder
- `src/synpop/utils.py` - utility helpers
- `src/synpop_model.py` - minimal runnable entrypoint
- `tests/test_structural_model.py` - structural tests for core dynamics

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

For direct imports without packaging, set:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## Quick Run

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m synpop_model
```

## Tests

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
pytest tests\test_structural_model.py -q
```
