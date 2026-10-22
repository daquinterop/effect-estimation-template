import copy
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data.base import BaseData
from src.estimators.base import BaseEstimator, CausalEstimate, DoseResponseMixin

logger = logging.getLogger("causal_pipeline")


class SLearner(BaseEstimator, DoseResponseMixin):
    """
    S-Learner (Single Learner) causal effect estimator with Random Forest.

    Treats the treatment as an ordinary feature alongside control covariates
    in a single regression model.

    Parameters
    ----------
    global_config : dict
        Global configuration dictionary containing keys such as 'treatment_node',
        'outcome_node', and 'random_seed'.
    exp_config : dict
        Experiment configuration dictionary.
    controls : list of str
        List of control variable names.
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    description = "S-Learner with Random Forest"

    def __init__(self, global_config: dict, exp_config: dict, controls: list[str], **kwargs):
        super().__init__(global_config, exp_config, controls, **kwargs)
        self._models = []
        self._base_learner = RandomForestRegressor(
            n_estimators=100,
            random_state=global_config["random_seed"],
            max_depth=5,
        )

    def _fit_single_instance(
        self, X: pd.DataFrame, T: pd.Series, Y: pd.Series
    ) -> None:
        """
        Fit a single Random Forest model on covariates concatenated with treatment.

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
        model = copy.deepcopy(self._base_learner)
        model.fit(pd.concat([X, T], axis=1), Y)
        self._models.append(model)

    def fit(self, data: BaseData, bootstrap_n: int = 100) -> None:
        """
        Fit the S-Learner estimator using bootstrap resampling.

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

    def estimate_ate(self) -> CausalEstimate:
        """
        Estimate Average Treatment Effect (ATE).

        Returns
        -------
        CausalEstimate
            Causal estimate of the average treatment effect.
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

    def predict(
        self, X: pd.DataFrame, T: pd.Series, model: RandomForestRegressor
    ) -> np.ndarray:
        """
        Predict outcome values for a given fitted Random Forest model.

        Parameters
        ----------
        X : pd.DataFrame
            Covariates dataframe.
        T : pd.Series
            Treatment series.
        model : RandomForestRegressor
            Fitted Random Forest regressor instance.

        Returns
        -------
        np.ndarray
            Predicted outcome values.
        """
        return model.predict(pd.concat([X, T], axis=1))

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