from abc import ABC, abstractmethod
from typing import Any

from src.estimators.base import CausalEstimate


class BaseRobustnessCheck(ABC):
    """
    Abstract Base Class for Robustness Checks.

    Takes a CausalEstimate object and re-runs or permutes the original
    estimator to test the sensitivity or robustness of the original result.

    Parameters
    ----------
    estimate : CausalEstimate
        The original CausalEstimate object, which holds a reference to the
        estimator and its data.
    **kwargs : dict, optional
        Additional parameters for the robustness check.

    Attributes
    ----------
    original_estimate : CausalEstimate
        The original causal estimate being checked.
    kwargs : dict
        Additional parameters passed to the check.
    results : Any
        The results produced by `run_check`.
    """

    def __init__(self, estimate: CausalEstimate, **kwargs):
        self.original_estimate = estimate
        self.kwargs = kwargs
        self.results = self.run_check()

    @abstractmethod
    def run_check(self) -> Any:
        """
        Execute the robustness check.

        Returns
        -------
        Any
            The results of the robustness check (e.g. the new effect size,
            a p-value for the check, etc.).
        """
        pass

