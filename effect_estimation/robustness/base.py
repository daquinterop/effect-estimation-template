from abc import ABC, abstractmethod
from typing import Any
from effect_estimation.estimators.base import CausalEstimate

class BaseRobustnessCheck(ABC):
    """
    Abstract Base Class for Robustness Checks.
    
    Takes a CausalEstimate object and re-runs or permutes the original 
    estimator to test the sensitivity or robustness of the original result.
    """
    
    def __init__(self, estimate: CausalEstimate, **kwargs):
        """
        Initialize the robustness check.
        
        Args:
            estimate: The original CausalEstimate object, which holds a 
                      reference to the estimator and its data.
            **kwargs: Additional parameters for the robustness check.
        """
        self.original_estimate = estimate
        self.kwargs = kwargs
        self.results = self.run_check()

    @abstractmethod
    def run_check(self) -> Any:
        """
        Execute the robustness check.
        
        Returns:
            The results of the robustness check (e.g. the new effect size,
            a p-value for the check, etc.).
        """
        pass
