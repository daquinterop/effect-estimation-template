from dowhy.causal_identifier import adjustment_set
from networkx.generators import spectral_graph_forge
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from src.data.base import BaseData

@dataclass
class CausalEstimate:
    """
    Dataclass to hold the results of a causal estimation.
    """
    value: float
    std_error: float
    p_value: float
    ci_lower: float
    ci_upper: float

class BaseEstimator(ABC):
    """
    Abstract Base Class for Causal Estimators.
    
    Instantiated directly with data (statsmodels style) to tightly couple
    the estimator instance to the dataset it operates on.
    """
    description: str | None = None
    def __init__(self, global_config: dict, exp_config: dict, controls: list[str], **kwargs):
        """
        Initialize the estimator with data.
        
        Args:
            outcome: The outcome variable.
            treatment: The treatment variable.
            controls: The control/confounder variables.
            conditional_nodes: Effect modifiers (W) used for CATE estimation.
            **kwargs: Additional estimator-specific arguments.
        """
        self.treatment = global_config.get("treatment_node")
        self.outcome = global_config.get("outcome_node")
        self.controls = controls
        self.conditional_nodes = global_config.get("conditional_nodes", [])
        self.kwargs = kwargs
        self.estimator = None
        self.data: BaseData = None
        self.treatment_features = None
    
    @abstractmethod
    def fit(self, data: BaseData, **kwargs) -> None:
        """
        Fit the estimator to the data.
        """
        pass

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
    def dose_response_curve(self, t: float) -> CausalEstimate:
        """
        Returns the conditional expectation of the outcome given the treatment level t,
        E[Y(t)]
        """
        pass