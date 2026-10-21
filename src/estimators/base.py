from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class CausalEstimate:
    """
    Dataclass to hold the results of a causal estimation.
    """
    value: float
    std_error: float
    p_value: float
    estimator_instance: Any  # Reference to the estimator that generated this
    ci_lower: float
    ci_upper: float

class BaseEstimator(ABC):
    """
    Abstract Base Class for Causal Estimators.
    
    Instantiated directly with data (statsmodels style) to tightly couple
    the estimator instance to the dataset it operates on.
    """
    
    def __init__(
        self, 
        outcome: Any, 
        treatment: Any, 
        controls: Any, 
        conditional_nodes: Optional[Any] = None, 
        splits: Optional[Any] = None, 
        **kwargs
    ):
        """
        Initialize the estimator with data.
        
        Args:
            outcome: The outcome variable.
            treatment: The treatment variable.
            controls: The control/confounder variables.
            conditional_nodes: Effect modifiers (W) used for CATE estimation.
            splits: Indices for cross-fitting (if applicable).
            **kwargs: Additional estimator-specific arguments.
        """
        self.outcome = outcome
        self.treatment = treatment
        self.controls = controls
        self.conditional_nodes = conditional_nodes
        self.splits = splits
        self.kwargs = kwargs
    
    @abstractmethod
    def estimate_ate(self) -> CausalEstimate:
        """
        Estimate the Average Treatment Effect (ATE).
        
        Returns:
            A CausalEstimate object containing the estimate, SE, p-value,
            and a reference to this estimator instance.
        """
        pass
        
    def estimate_cate(self) -> Any:
        """
        Estimate the Conditional Average Treatment Effect (CATE).
        This can be implemented by subclasses that support heterogeneous effects.
        """
        raise NotImplementedError("CATE estimation is not implemented for this estimator.")

class DoseResponseMixin:
    """
    A class to implement dose-response methods
    """
    @abstractmethod
    def dose_response_curve(self, t: float):
        """
        Returns the conditional expectation of the outcome given the treatment level t,
        E[Y(t)]
        """
        pass