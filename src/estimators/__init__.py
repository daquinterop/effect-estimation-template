"""
Estimators for causal effects.
"""
from typing import Type

from src.estimators.base import BaseEstimator
from src.estimators.gps import GPS
from src.estimators.meta_learners import SLearner
from src.estimators.ols import LinearOLS, LinearOLSPoly

ESTIMATORS = {
    "LinearOLS": LinearOLS,
    "LinearOLSPoly": LinearOLSPoly,
    "GPS": GPS,
    "SLearner": SLearner,
}


def get_estimator(estimator_name: str) -> Type[BaseEstimator]:
    """
    Retrieve an estimator class by its registered string name.

    Parameters
    ----------
    estimator_name : str
        The name of the causal estimator (e.g., 'LinearOLSPoly', 'GPS', 'SLearner').

    Returns
    -------
    Type[BaseEstimator]
        The estimator class corresponding to `estimator_name`.

    Raises
    ------
    ValueError
        If `estimator_name` is not found in the registered estimators mapping.
    """
    if estimator_name not in ESTIMATORS:
        raise ValueError(f"Estimator '{estimator_name}' not found.")
    return ESTIMATORS[estimator_name]