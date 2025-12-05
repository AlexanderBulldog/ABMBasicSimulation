"""
Compatibility shim for the synpop package.

Exports EconomyModel and scenario helpers from synpop.* modules.
"""

from synpop import EconomyModel, run_demo, run_scenarios, summarize_run


if __name__ == "__main__":
    run_demo()
