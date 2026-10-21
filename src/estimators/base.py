from dowhy.causal_identifier import adjustment_set
from networkx.generators import spectral_graph_forge
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from scipy import stats
import statsmodels.api as sm

import pandas as pd
import numpy as np

from src.data.base import BaseData
import logging

logger = logging.getLogger("causal_pipeline")

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
        self.random_seed = global_config.get('random_seed')
        self.kwargs = kwargs
        self.estimator = None # To save single model instance when not boostrapping
        self.data: BaseData = None
        self.treatment_features = None
        self._models = None # To save the model instances in bootstraping
    
    @abstractmethod
    def fit(self, data: BaseData, **kwargs) -> None:
        """
        Fit the estimator to the data.
        """
        pass

    def _fit_single_instance(self, X, T, Y):
        """
        Necesary for flexible bootstrap implementation. Here is an example of how that
        would work
        """
        model, _, _, _ = np.linalg.lstsq(sm.add_constant(np.hstack([X, T])), Y, rcond=None)
        self._models.append(model)
    
    def _boostrap_fit(self, data: BaseData, bootstrap_n: int = 100, **kwargs):
        """
        A default fit bootstrap implementation
        """
        logger.info(f"Boostrap fitting {self.description}")
        self.data = data.data.copy()
        for i in range(bootstrap_n):
            bootstrapped_data = self.data.sample(n=len(self.data), replace=True, random_state=self.random_seed + 27*i)
            self._fit_single_instance(
                X = bootstrapped_data[self.controls], 
                T = bootstrapped_data[self.treatment], 
                Y = bootstrapped_data[self.outcome]
            )

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
    def conditional_expectation(self, t: float) -> CausalEstimate:
        """
        Returns the conditional expectation of the outcome given the treatment level t,
        E[Y(t) | controls]
        """
        pass

    def bootstrap_conditional_expectation(self, t: float) -> CausalEstimate:
        """
        A method for bootstrap inference. Necessary for some estimators
        """
        X = self.data[self.controls]
        point_estimates = []
        for model in self._models:
            T = pd.Series([t] * len(X), index=X.index, name=self.treatment)
            predictions = self.predict(X, T)
            point_estimates.append(np.mean(predictions))
        value = np.mean(point_estimates)
        se = np.std(point_estimates)
        estimate = CausalEstimate(
            value=value, 
            std_error=se, 
            p_value=2*min(stats.norm.cdf(0, value, se), stats.norm.sf(0, value, se)), 
            ci_lower=value - 1.96*se, 
            ci_upper=value + 1.96*se,
        )
        return estimate
    
    @abstractmethod
    def dose_response_curve(self, t_grid: list[float]) -> pd.DataFrame:
        """
        Returns the dose-response curve
        """
        pass

    def bootstrap_dose_response(self, t_grid: list[float]) -> pd.DataFrame:
        """
        Boostraps the dose-response curve
        """
        df_list = []
        for t_i in t_grid:
            tmp_df = self.data[self.controls].copy()
            tmp_df[self.treatment] = t_i
            df_list.append(tmp_df)
        df = pd.concat(df_list)

        response_df = pd.DataFrame(
            columns=["estimate", "std_error", "p_value", "ci_lower", "ci_upper", "t_value"], 
            index=t_grid
        )

        T = df[self.treatment]
        X = df.drop(columns=self.treatment)

        point_estimates = {
            "estimate": [],
            "t_value": [],
            "bootstrap_id": []
        }
        for i, (model) in enumerate(self._models):
            predictions = self.predict(X, T, model)
            point_estimates["estimate"].extend(predictions)
            point_estimates["t_value"].extend(T.to_numpy())
            point_estimates["bootstrap_id"].extend([i]*len(predictions))
        
        point_estimates = pd.DataFrame(point_estimates)
        response_df['estimate'] = point_estimates.groupby('t_value')['estimate'].mean()
        response_df['std_error'] = point_estimates.groupby(['t_value', "bootstrap_id"])['estimate'].mean().groupby(level=0).std()
        response_df['p_value'] = response_df.apply(
            lambda row: 2*min(stats.norm.cdf(0, row.estimate, row.std_error), stats.norm.sf(0, row.estimate, row.std_error)),
            axis=1
        )
        response_df['ci_lower'] = response_df.estimate - 1.96*response_df.std_error
        response_df['ci_upper'] = response_df.estimate + 1.96*response_df.std_error
        return response_df.drop(columns='t_value')

    
    def predict(self, T, X, model):
        """
        It must retun the point predictions of the model as any sklearn Predictor would
        do it
        """
        pass
