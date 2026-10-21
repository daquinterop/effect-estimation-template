"""
Estimators for causal effects.
"""
from src.estimators.ols import LinearOLS, LinearOLSPoly, BaseEstimator

ESTIMATORS = {
    "LinearOLS": LinearOLS,
    "LinearOLSPoly": LinearOLSPoly
}

def get_estimator(estimator_name) -> BaseEstimator:
    if estimator_name not in ESTIMATORS:
        raise ValueError(f"Estimator '{estimator_name}' not found.")
    return ESTIMATORS[estimator_name]