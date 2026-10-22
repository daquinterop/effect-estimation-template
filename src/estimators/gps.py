import logging

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from src.data.base import BaseData
from src.estimators.base import BaseEstimator, CausalEstimate, DoseResponseMixin

logger = logging.getLogger("causal_pipeline")


class GPS(BaseEstimator, DoseResponseMixin):
    """
    Hirano & Imbens (2004) Generalized Propensity Score (GPS) estimator.

    Designed for continuous treatments using a two-stage regression approach
    with bootstrap inference.

    Parameters
    ----------
    global_config : dict
        Global configuration dictionary.
    exp_config : dict
        Experiment configuration dictionary.
    controls : list of str
        List of control variable names.
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    description = "Hirano Generalized propensity score"

    def __init__(self, global_config: dict, exp_config: dict, controls: list[str], **kwargs):
        super().__init__(global_config, exp_config, controls, **kwargs)
        self._models = []

    def _conditional_expectation_featurizer(
        self, T: np.ndarray | pd.Series, R_hat: np.ndarray | pd.Series
    ) -> np.ndarray:
        """
        Construct polynomial feature matrix of treatment and estimated GPS.

        Parameters
        ----------
        T : np.ndarray or pd.Series
            Treatment array.
        R_hat : np.ndarray or pd.Series
            Estimated generalized propensity scores.

        Returns
        -------
        np.ndarray
            Design matrix with constant, linear, quadratic, and interaction terms.
        """
        N = len(T)
        return np.vstack([
            np.ones(N),
            T,
            T**2,
            R_hat,
            R_hat**2,
            T * R_hat,
        ]).T

    def _r_hat_estimate(
        self,
        X: pd.DataFrame,
        T: pd.Series,
        beta_hat: np.ndarray,
        sigma_sq_hat: float,
    ) -> np.ndarray:
        """
        Compute generalized propensity score density estimate R_hat.

        Parameters
        ----------
        X : pd.DataFrame
            Covariates dataframe.
        T : pd.Series
            Treatment series.
        beta_hat : np.ndarray
            Estimated first-stage regression coefficients.
        sigma_sq_hat : float
            Estimated residual variance of the treatment model.

        Returns
        -------
        np.ndarray
            Estimated propensity densities.
        """
        R_hat = (1 / np.sqrt(2 * np.pi * sigma_sq_hat)) * np.exp(
            -(T - sm.add_constant(X) @ beta_hat) ** 2 / (2 * sigma_sq_hat)
        )
        return R_hat


    def _fit_single_instance(
        self, X: pd.DataFrame, T: pd.Series, Y: pd.Series
    ) -> None:
        r"""
        Fit the GPS estimator on a single bootstrap resample.

        In the first stage, treatment is modeled conditionally normal:
        .. math::
            T_i \mid X_i \sim \mathcal{N}(\beta_0 + \beta_1 X_i, \sigma^2)

        In the second stage, outcome expectation is modeled as:
        .. math::
            E[Y_i \mid T_i, R_i] = \alpha_0 + \alpha_1 T_i + \alpha_2 T_i^2
            + \alpha_3 R_i + \alpha_4 R_i^2 + \alpha_5 T_i R_i

        Parameters
        ----------
        X : pd.DataFrame
            Covariates dataframe.
        T : pd.Series
            Treatment series.
        Y : pd.Series
            Outcome series.

        Returns
        -------
        None
        """
        beta_hat, _, _, _ = np.linalg.lstsq(sm.add_constant(X), T, rcond=None)
        residuals = T - sm.add_constant(X) @ beta_hat
        N = len(T)
        sigma_sq_hat = np.sum(residuals**2) / max(1, (N - X.shape[1]))

        R_hat = self._r_hat_estimate(X, T, beta_hat, sigma_sq_hat)

        T_R_features = self._conditional_expectation_featurizer(T, R_hat)
        alpha_hat, _, _, _ = np.linalg.lstsq(T_R_features, Y, rcond=None)

        self._models.append((beta_hat, sigma_sq_hat, alpha_hat))

    def fit(self, data: BaseData, bootstrap_n: int = 100) -> None:
        """
        Fit the estimator using bootstrap resampling.

        Parameters
        ----------
        data : BaseData
            Data container containing the dataset to fit.
        bootstrap_n : int, default=100
            Number of bootstrap samples.

        Returns
        -------
        None
        """
        super().fit(data)
        self._boostrap_fit(data, bootstrap_n)

    def _boostrap_fit(self, data: BaseData, bootstrap_n: int = 100, **kwargs):
        """
        Execute bootstrap fitting loop.

        Parameters
        ----------
        data : BaseData
            Data container containing the dataset to fit.
        bootstrap_n : int, default=100
            Number of bootstrap samples.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        None
        """
        return super()._boostrap_fit(data, bootstrap_n, **kwargs)

    def _treatment_featurizer(self, t_series: pd.Series) -> pd.DataFrame:
        """
        Format treatment series into design dataframe.

        Parameters
        ----------
        t_series : pd.Series
            Treatment series.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the treatment column.
        """
        df_t = pd.DataFrame({self.treatment: t_series})
        self.treatment_features = df_t.columns.to_list()
        return df_t

    def estimate_ate(self) -> CausalEstimate:
        """
        Estimate Average Treatment Effect (ATE).

        Returns
        -------
        CausalEstimate
            Causal estimate of average treatment effect.
        """
        raise NotImplementedError

    def estimate_cate(self) -> CausalEstimate:
        """
        Estimate Conditional Average Treatment Effect (CATE).

        Returns
        -------
        CausalEstimate
            Causal estimate across effect modifiers.
        """
        raise NotImplementedError

    def conditional_expectation(self, t: float) -> CausalEstimate:
        """
        Compute conditional expectation E[Y(t)] at treatment level t.

        Parameters
        ----------
        t : float
            Treatment dose level.

        Returns
        -------
        CausalEstimate
            Estimate and bootstrap confidence interval.
        """
        return self.bootstrap_conditional_expectation(t)

    def bootstrap_conditional_expectation(self, t: float) -> CausalEstimate:
        """
        Evaluate bootstrap conditional expectation across saved models.

        Parameters
        ----------
        t : float
            Treatment dose level.

        Returns
        -------
        CausalEstimate
            Estimate and bootstrap confidence interval.
        """
        return super().bootstrap_conditional_expectation(t)

    def predict(self, X: pd.DataFrame, T: pd.Series, model: tuple) -> np.ndarray:
        """
        Predict outcome values for a given bootstrap model triple.

        Parameters
        ----------
        X : pd.DataFrame
            Covariates dataframe.
        T : pd.Series
            Treatment series.
        model : tuple of (np.ndarray, float, np.ndarray)
            Tuple of (beta_hat, sigma_sq_hat, alpha_hat).

        Returns
        -------
        np.ndarray
            Array of predicted outcome values.
        """
        beta_hat, sigma_sq_hat, alpha_hat = model
        R_hat = self._r_hat_estimate(X, T, beta_hat, sigma_sq_hat)
        T_R_features = self._conditional_expectation_featurizer(T, R_hat)
        predictions = T_R_features @ alpha_hat
        return predictions

    def dose_response_curve(self, t_grid: list[float]) -> pd.DataFrame:
        """
        Compute dose-response curve across a grid of treatment doses.

        Parameters
        ----------
        t_grid : list of float
            Array of treatment values.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by treatment level containing estimates and CIs.
        """
        return self.bootstrap_dose_response(t_grid)

    def bootstrap_dose_response(self, t_grid: list[float]) -> pd.DataFrame:
        """
        Compute bootstrap dose-response curve.

        Parameters
        ----------
        t_grid : list of float
            Array of treatment values.

        Returns
        -------
        pd.DataFrame
            DataFrame containing bootstrap estimates, standard errors, and CIs.
        """
        return super().bootstrap_dose_response(t_grid)

        
            