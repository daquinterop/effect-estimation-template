from src.estimators.base import BaseEstimator, DoseResponseMixin, CausalEstimate
from src.data.base import BaseData

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import logging

logger = logging.getLogger("causal_pipeline")

class GPS(BaseEstimator, DoseResponseMixin):
    """
    Hirano & Imbens (2004) GPS estimator for continuous treatments
    """
    description = "Hirano Generalized propsensity score"
    def __init__(self, global_config, exp_config, controls, **kwargs):
        super().__init__(global_config, exp_config, controls, **kwargs)
        self._models = []

    def _conditional_expectation_featurizer(self, T, R_hat):
        N = len(T)
        return np.vstack([
            np.ones(N),
            T,
            T**2,
            R_hat,
            R_hat**2,
            T * R_hat
        ]).T

    def _r_hat_estimate(self, T, X, beta_hat, sigma_sq_hat):
        R_hat = (1/np.sqrt(2*np.pi*sigma_sq_hat)) * \
            np.exp(-(T - sm.add_constant(X) @ beta_hat)**2 / (2*sigma_sq_hat))
        return R_hat

    def _fit_single_instance(self, X, T, Y):
        """
        It fits the estimator for a single instance.
        During the first stage the treatment is approximated as 
           T_i|X_i ~ N(\beta_0 + \beta_1 * X_i + \sigma_i). 

        Then, the conditional density function is defined as:
           R(t|x) = \frac{1}{\sqrt{2\pi \sigma^2}} exp{-\frac{(t - (\beta_0 + \beta_1 * X))^2}{2\sigma^2}}
        
        In the second stage, the conditional expectation of Y_i given T_i and R_i is modeled as:
           E[Y_i | T_i, R_i] = \alpha_0 + \alpha_1 * T_i + \alpha_2 * T_i^2 + \alpha_3 * R_i + \alpha_4 * R_i^2 + \alpha_5 * T_i * R_i
        
        """
        beta_hat, _, _, _ = np.linalg.lstsq(sm.add_constant(X), T, rcond=None)
        residuals = T - sm.add_constant(X) @ beta_hat
        N = len(T)
        sigma_sq_hat = np.sum(residuals**2) / max(1, (N - X.shape[1]))

        R_hat = self._r_hat_estimate(T, X, beta_hat, sigma_sq_hat)

        T_R_features = self._conditional_expectation_featurizer(T, R_hat)
        alpha_hat, _, _, _ = np.linalg.lstsq(T_R_features, Y, rcond=None)

        self._models.append((beta_hat, sigma_sq_hat, alpha_hat))
        
    def fit(self, data: BaseData, bootstrap_n: int = 100) -> None:
        """
        Boostraps the fitting of the estimator.
        """
        self.data = data.data.copy()
        self._boostrap_fit(data, bootstrap_n)

    def _boostrap_fit(self, data: BaseData, bootstrap_n: int = 100, **kwargs):
        return super()._boostrap_fit(data, bootstrap_n, **kwargs)

    def _treatment_featurizer(self, t_series):
        df_t = pd.DataFrame({self.treatment: t_series})
        self.treatment_features = df_t.columns.to_list()
        return df_t

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
        return self.bootstrap_conditional_expectation(t)

    def bootstrap_conditional_expectation(self, t: float) -> CausalEstimate:
        return super().bootstrap_conditional_expectation(t)

    def predict(self, X, T, model):
        beta_hat, sigma_sq_hat, alpha_hat = model
        R_hat = self._r_hat_estimate(T, X, beta_hat, sigma_sq_hat)
        T_R_features = self._conditional_expectation_featurizer(T, R_hat)
        predictions = sm.add_constant(T_R_features) @ alpha_hat
        return predictions

    def dose_response_curve(self, t_grid: list[float]) -> pd.DataFrame:
        return self.bootstrap_dose_response(t_grid)

    def bootstrap_dose_response(self, t_grid: list[float]) -> pd.DataFrame:
        return super().bootstrap_dose_response(t_grid)
        
            