"""
Compatibility shim for the synpop package.

Exports EconomyModel from synpop.* modules.
"""

from synpop import EconomyModel


if __name__ == "__main__":
    model = EconomyModel(seed=1, n_households=80, n_firms=8, enable_credit=True, price_elasticity=2.0)
    model.run_model(steps=50)
    print(model.results_dataframe().tail(5))
