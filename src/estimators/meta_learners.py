from src.estimators.base import BaseEstimator, DoseResponseMixin, CausalEstimate
from src.data.base import BaseData
from sklearn.ensemble import RandomForestRegressor
import copy
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger("causal_pipeline")

class SLearner(BaseEstimator, DoseResponseMixin):
    description = "S-Learner with Random Forest"
    def __init__(self, global_config, exp_config, controls, **kwargs):
        super().__init__(global_config, exp_config, controls, **kwargs)
        self._models = []
        self._base_learner = RandomForestRegressor(
            n_estimators=100, random_state=global_config["random_seed"]
        )
    
    def _fit_single_instance(self, X, T, Y):
        """
        Fits the S-Learner to the data for a single instance.
        """
        model = copy.deepcopy(self._base_learner)       
        model.fit(pd.concat([X, T], axis=1), Y)
        self._models.append(model)        

    def fit(self, data: BaseData, bootstrap_n: int = 100) -> None:
        """
        Boostraps the fitting of the estimator.
        """
        self.data = data.data.copy()
        self._boostrap_fit(data, bootstrap_n)

    def _boostrap_fit(self, data: BaseData, bootstrap_n: int = 100, **kwargs):
        return super()._boostrap_fit(data, bootstrap_n, **kwargs)
    
    def estimate_ate(self) -> CausalEstimate:
        raise NotImplementedError
    
    def estimate_cate(self) -> CausalEstimate:
        raise NotImplementedError
    
    def conditional_expectation(self, t: float) -> CausalEstimate:
        """
        Calculates the dose response curve at a given value of t. This is the conditional expectation 
        of the outcome given the treatment level t, E[Y(t)].

        Inference is conducted using the bootstrap samples of the estimated parameters.
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

    def predict(self, X, T, model):
        return model.predict(pd.concat([X, T], axis=1))

    def dose_response_curve(self, t_grid: list[float]) -> pd.DataFrame:
        return self.bootstrap_dose_response(t_grid)

    def bootstrap_dose_response(self, t_grid: list[float]) -> pd.DataFrame:
        return super().bootstrap_dose_response(t_grid)