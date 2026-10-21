"""
Estimators for causal effects.
"""
from src.estimators.ols import LinearOLS, LinearOLSPoly, BaseEstimator
from src.estimators.gps import GPS
from src.estimators.meta_learners import SLearner

ESTIMATORS = {
    "LinearOLS": LinearOLS,
    "LinearOLSPoly": LinearOLSPoly,
    "GPS": GPS,
    "SLearner": SLearner
}

def get_estimator(estimator_name) -> BaseEstimator:
    if estimator_name not in ESTIMATORS:
        raise ValueError(f"Estimator '{estimator_name}' not found.")
    return ESTIMATORS[estimator_name]