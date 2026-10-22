import pandas as pd
from scipy import stats
import statsmodels.api as sm

from src.data.base import BaseData
from src.estimators.base import BaseEstimator, CausalEstimate, DoseResponseMixin


class LinearOLS(BaseEstimator, DoseResponseMixin):
    """
    Linear OLS estimator for Average Treatment Effect (ATE).

    Models outcome linearly as a function of treatment and controls.
    Does not model effect heterogeneity.

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

    description = "Linear OLS"

    def __init__(self, global_config: dict, exp_config: dict, controls: list[str], **kwargs):
        super().__init__(global_config, exp_config, controls, **kwargs)
        self.endog = None
        self.exog = None

    def fit(self, data: BaseData) -> None:
        """
        Fit the OLS regression model on treatment and control covariates.

        Parameters
        ----------
        data : BaseData
            Data container containing the dataset to fit.

        Returns
        -------
        None
        """
        super().fit(data)
        self.exog = self._get_exog(self.data[self.treatment])
        self.endog = self.data[self.outcome]
        self.estimator = sm.OLS(endog=self.endog, exog=self.exog).fit()

    def _treatment_featurizer(self, t_series: pd.Series) -> pd.DataFrame:
        """
        Featurize treatment series into a design matrix dataframe.

        Parameters
        ----------
        t_series : pd.Series
            Series of treatment values.

        Returns
        -------
        pd.DataFrame
            DataFrame containing featurized treatment variables.
        """
        df_t = pd.DataFrame({self.treatment: t_series})
        self.treatment_features = df_t.columns.to_list()
        return df_t

    def _get_exog(self, t_series: pd.Series) -> pd.DataFrame:
        """
        Construct design matrix combining control covariates and featurized treatment.

        Parameters
        ----------
        t_series : pd.Series
            Series of treatment values.

        Returns
        -------
        pd.DataFrame
            Design matrix with added constant.
        """
        exog = pd.concat([
            self.data.loc[:, self.controls],
            self._treatment_featurizer(t_series)
        ], axis=1)
        exog = sm.add_constant(exog, has_constant="add")
        return exog

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
        Compute conditional expectation E[Y(t) | controls] at treatment level t.

        Parameters
        ----------
        t : float
            Treatment dose level.

        Returns
        -------
        CausalEstimate
            Predicted expectation and analytical confidence interval.
        """
        exog = self._get_exog(pd.Series([t] * len(self.data), index=self.data.index))
        prediction = self.estimator.get_prediction(exog.mean().to_frame().T)
        pred_val = prediction.predicted.item()
        pred_se = prediction.se.item()
        p_val = 2 * min(
            stats.norm.cdf(0, pred_val, pred_se),
            stats.norm.sf(0, pred_val, pred_se),
        )
        return CausalEstimate(
            value=pred_val,
            std_error=pred_se,
            p_value=p_val,
            ci_lower=pred_val - 1.96 * pred_se,
            ci_upper=pred_val + 1.96 * pred_se,
        )

    def dose_response_curve(self, t_grid: list[float]) -> pd.DataFrame:
        """
        Compute dose-response curve E[Y(t)] across treatment grid values.

        Parameters
        ----------
        t_grid : list of float
            Grid of treatment levels to evaluate.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by treatment dose with estimates and confidence intervals.
        """
        response_df = pd.DataFrame(
            columns=["estimate", "std_error", "p_value", "ci_lower", "ci_upper"],
            index=t_grid,
        )
        for t in t_grid:
            exog = self._get_exog(pd.Series([t] * len(self.data), index=self.data.index))
            prediction = self.estimator.get_prediction(exog.mean().to_frame().T)
            pred_val = prediction.predicted.item()
            pred_se = prediction.se.item()
            p_val = 2 * min(
                stats.norm.cdf(0, pred_val, pred_se),
                stats.norm.sf(0, pred_val, pred_se),
            )
            response_df.loc[t] = (
                pred_val,
                pred_se,
                p_val,
                pred_val - 1.96 * pred_se,
                pred_val + 1.96 * pred_se,
            )
        return response_df


class LinearOLSPoly(LinearOLS):
    """
    Linear OLS estimator with quadratic polynomial treatment specification.

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

    description = "Linear OLS with Polynomial Treatment"

    def __init__(self, global_config: dict, exp_config: dict, controls: list[str], **kwargs):
        super().__init__(global_config, exp_config, controls, **kwargs)

    def _treatment_featurizer(self, t_series: pd.Series) -> pd.DataFrame:
        """
        Featurize treatment into linear and quadratic terms (t, t^2).

        Parameters
        ----------
        t_series : pd.Series
            Series of treatment values.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 't' and 't2'.
        """
        df_t = pd.DataFrame(index=self.data.index)
        df_t["t"] = t_series
        df_t["t2"] = t_series ** 2
        self.treatment_features = df_t.columns.to_list()
        return df_t

    def fit(self, data: BaseData) -> None:
        """
        Fit polynomial OLS model to the data.

        Parameters
        ----------
        data : BaseData
            Data container containing the dataset to fit.

        Returns
        -------
        None
        """
        super().fit(data)

    