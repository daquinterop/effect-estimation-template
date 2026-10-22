from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from src.data.base import BaseData

logger = logging.getLogger("causal_pipeline")


@dataclass
class CausalEstimate:
    """
    Dataclass to hold the results of a causal estimation.

    Parameters
    ----------
    value : float
        Point estimate value.
    std_error : float
        Standard error of the estimate.
    p_value : float
        Two-sided p-value against the null hypothesis of zero effect.
    ci_lower : float
        Lower bound of the 95% confidence interval.
    ci_upper : float
        Upper bound of the 95% confidence interval.
    """

    value: float
    std_error: float
    p_value: float
    ci_lower: float
    ci_upper: float


class BaseEstimator(ABC):
    """
    Abstract Base Class for Causal Estimators.

    Instantiated directly with data to couple the estimator instance to the
    dataset it operates on.

    Parameters
    ----------
    global_config : dict
        Global configuration dictionary containing keys such as 'treatment_node',
        'outcome_node', 'conditional_nodes', and 'random_seed'.
    exp_config : dict
        Experiment configuration dictionary.
    controls : list of str
        List of control/confounder column names.
    **kwargs : dict, optional
        Additional estimator-specific parameters.

    Attributes
    ----------
    treatment : str
        Name of the treatment variable.
    outcome : str
        Name of the outcome variable.
    controls : list of str
        List of control variable names.
    conditional_nodes : list
        Effect modifiers used for CATE estimation.
    random_seed : int or None
        Random seed for reproducibility.
    data : BaseData or None
        Subset of data used for estimation.
    """

    description: str | None = None

    def __init__(
        self, global_config: dict, exp_config: dict, controls: list[str], **kwargs
    ):
        self.treatment = global_config.get("treatment_node")
        self.outcome = global_config.get("outcome_node")
        self.controls = controls
        self.conditional_nodes = global_config.get("conditional_nodes", [])
        self.random_seed = global_config.get("random_seed")
        self.kwargs = kwargs
        self.estimator = None  # Single model instance when not bootstrapping
        self.data: BaseData = None
        self.treatment_features = None
        self._models = None  # Model instances in bootstrapping

    @abstractmethod
    def fit(self, data: BaseData, **kwargs) -> None:
        """
        Fit the estimator to the data.

        Parameters
        ----------
        data : BaseData
            Data container containing the dataset to fit.
        **kwargs : dict, optional
            Additional fitting parameters.

        Returns
        -------
        None
        """
        self.data = data.data.copy()
        self.data = self.data[self.controls + [self.treatment, self.outcome]]

    def _fit_single_instance(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> None:
        """
        Fit a single model instance on given covariates, treatment, and outcome.

        Parameters
        ----------
        X : pd.DataFrame
            Control covariates dataframe.
        T : pd.Series
            Treatment series.
        Y : pd.Series
            Outcome series.

        Returns
        -------
        None
        """
        model, _, _, _ = np.linalg.lstsq(
            sm.add_constant(np.hstack([X, T])), Y, rcond=None
        )
        self._models.append(model)

    def _boostrap_fit(
        self, data: BaseData, bootstrap_n: int = 100, **kwargs
    ) -> None:
        """
        Fit the estimator using bootstrap resampling.

        Parameters
        ----------
        data : BaseData
            Data container containing the dataset to sample from.
        bootstrap_n : int, default=100
            Number of bootstrap iterations.
        **kwargs : dict, optional
            Additional fitting parameters.

        Returns
        -------
        None
        """
        logger.info(f"Bootstrap fitting {self.description}")
        self.data = data.data.copy()
        for i in range(bootstrap_n):
            bootstrapped_data = self.data.sample(
                n=len(self.data), replace=True, random_state=self.random_seed + 27 * i
            )
            self._fit_single_instance(
                X=bootstrapped_data[self.controls],
                T=bootstrapped_data[self.treatment],
                Y=bootstrapped_data[self.outcome],
            )

    @abstractmethod
    def estimate_ate(self) -> CausalEstimate:
        """
        Estimate the Average Treatment Effect (ATE).

        Returns
        -------
        CausalEstimate
            Object containing the estimate, standard error, p-value, and confidence intervals.
        """
        pass

    def estimate_cate(self) -> Any:
        """
        Estimate the Conditional Average Treatment Effect (CATE).

        Returns
        -------
        Any
            Estimated conditional treatment effects across effect modifiers.

        Raises
        ------
        NotImplementedError
            If CATE estimation is not implemented for this estimator.
        """
        raise NotImplementedError("CATE estimation is not implemented for this estimator.")


class DoseResponseMixin(ABC):
    """
    Mixin class implementing dose-response and treatment efficiency methods.
    """

    @abstractmethod
    def conditional_expectation(self, t: float) -> CausalEstimate:
        """
        Compute conditional expectation E[Y(t) | controls] at treatment level t.

        Parameters
        ----------
        t : float
            Treatment dose level.

        Returns
        -------
        CausalEstimate
            Estimated expectation and confidence interval.
        """
        pass

    def bootstrap_conditional_expectation(self, t: float) -> CausalEstimate:
        """
        Compute conditional expectation across bootstrap model samples.

        Parameters
        ----------
        t : float
            Treatment dose level.

        Returns
        -------
        CausalEstimate
            Estimated expectation and bootstrap standard error.
        """
        X = self.data[self.controls]
        point_estimates = []
        for model in self._models:
            T = pd.Series([t] * len(X), index=X.index, name=self.treatment)
            predictions = self.predict(X, T, model)
            point_estimates.append(np.mean(predictions))
        value = np.mean(point_estimates)
        se = np.std(point_estimates)
        estimate = CausalEstimate(
            value=value,
            std_error=se,
            p_value=2
            * min(
                stats.norm.cdf(0, value, se),
                stats.norm.sf(0, value, se),
            ),
            ci_lower=value - 1.96 * se,
            ci_upper=value + 1.96 * se,
        )
        return estimate

    @abstractmethod
    def dose_response_curve(self, t_grid: list[float]) -> pd.DataFrame:
        """
        Compute the dose-response curve across a grid of treatment values.

        Parameters
        ----------
        t_grid : list of float
            Array or list of treatment levels to evaluate.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by treatment level containing estimates, std errors, and CIs.
        """
        pass

    def average_treatment_efficiency(self) -> CausalEstimate:
        """
        Estimate factual treatment efficiency, the dose-normalized ATT.

        Defined as: E[(Y(T) - Y(0)) | T > 0] / E[T | T > 0].

        Returns
        -------
        CausalEstimate
            Efficiency estimate with standard error and confidence intervals.
        """
        Y_mean = self.data[self.outcome].mean()
        Y_se = self.data[self.outcome].std() / np.sqrt(len(self.data))
        T_mean = self.data[self.treatment].mean()

        expY_T0 = self.conditional_expectation(t=0)

        eff_value = (Y_mean - expY_T0.value) / T_mean
        eff_se = (Y_se**2 + expY_T0.std_error**2) ** 0.5 / T_mean
        efficiency = CausalEstimate(
            value=eff_value,
            std_error=eff_se,
            p_value=2
            * min(
                stats.norm.cdf(0, eff_value, eff_se),
                stats.norm.sf(0, eff_value, eff_se),
            ),
            ci_lower=eff_value - 1.96 * eff_se,
            ci_upper=eff_value + 1.96 * eff_se,
        )
        return efficiency

    def bootstrap_dose_response(self, t_grid: list[float]) -> pd.DataFrame:
        """
        Compute dose-response curve using predictions across bootstrap models.

        Parameters
        ----------
        t_grid : list of float
            Grid of treatment levels.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'estimate', 'std_error', 'p_value', 'ci_lower', 'ci_upper'.
        """
        df_list = []
        for t_i in t_grid:
            tmp_df = self.data[self.controls].copy()
            tmp_df[self.treatment] = t_i
            df_list.append(tmp_df)
        df = pd.concat(df_list)

        response_df = pd.DataFrame(
            columns=[
                "estimate",
                "std_error",
                "p_value",
                "ci_lower",
                "ci_upper",
                "t_value",
            ],
            index=t_grid,
        )

        T = df[self.treatment]
        X = df.drop(columns=self.treatment)

        point_estimates = {"estimate": [], "t_value": [], "bootstrap_id": []}
        for i, model in enumerate(self._models):
            predictions = self.predict(X, T, model)
            point_estimates["estimate"].extend(predictions)
            point_estimates["t_value"].extend(T.to_numpy())
            point_estimates["bootstrap_id"].extend([i] * len(predictions))

        point_estimates = pd.DataFrame(point_estimates)
        response_df["estimate"] = point_estimates.groupby("t_value")["estimate"].mean()
        response_df["std_error"] = (
            point_estimates.groupby(["t_value", "bootstrap_id"])["estimate"]
            .mean()
            .groupby(level=0)
            .std()
        )
        response_df["p_value"] = response_df.apply(
            lambda row: 2
            * min(
                stats.norm.cdf(0, row.estimate, row.std_error),
                stats.norm.sf(0, row.estimate, row.std_error),
            ),
            axis=1,
        )
        response_df["ci_lower"] = response_df.estimate - 1.96 * response_df.std_error
        response_df["ci_upper"] = response_df.estimate + 1.96 * response_df.std_error
        return response_df.drop(columns="t_value")

    def predict(self, X: pd.DataFrame, T: pd.Series, model: Any) -> np.ndarray:
        """
        Predict point estimates given covariates and treatment for a specific model instance.

        Parameters
        ----------
        X : pd.DataFrame
            Covariates dataframe.
        T : pd.Series
            Treatment series.
        model : Any
            Fitted model instance.

        Returns
        -------
        np.ndarray
            Array of predicted values.
        """
        pass

