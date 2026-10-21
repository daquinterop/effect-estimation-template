from src.estimators.base import BaseEstimator, DoseResponseMixin, CausalEstimate
from src.data.base import BaseData

import pandas as pd
import statsmodels.api as sm
from scipy import stats

class LinearOLS(BaseEstimator, DoseResponseMixin):
    """
    This is a basic OLS implementation for ATE. No effect heterogeneity (conditional nodes)
    are considered
    """
    description = "Linear OLS"
    def __init__(self, global_config, exp_config, controls, **kwargs):
        super().__init__(global_config, exp_config, controls, **kwargs)
        self.endog = None
        self.exog = None

    def fit(self, data: BaseData) -> None:
        self.data = data.data.copy()
        self.exog = self._get_exog(self.data[self.treatment])
        self.endog = self.data[self.outcome]
        self.estimator = sm.OLS(endog=self.endog, exog=self.exog).fit()

    def _treatment_featurizer(self, t_series):
        df_t = pd.DataFrame({self.treatment: t_series})
        self.treatment_features = df_t.columns.to_list()
        return df_t

    def _get_exog(self, t_series):
        """
        Get the covariates dataframe with the featurized treatment for the given
        treatment in t_series
        """
        exog = pd.concat([
            self.data.loc[:, self.controls],
            self._treatment_featurizer(t_series)
        ], axis=1)
        exog = sm.add_constant(exog, has_constant="add")
        return exog

    def estimate_ate(self) -> CausalEstimate:
        raise NotImplementedError
    
    def estimate_cate(self) -> CausalEstimate:
        raise NotImplementedError
    
    def dose_response_curve(self, t_grid: list[float]) -> pd.DataFrame:
        """
        Calculates the dose response curve at a given value of t. This is the conditional expectation 
        of the outcome given the treatment level t, E[Y(t)].
        """
        response_df = pd.DataFrame(
            columns=["estimate", "std_error", "p_value", "ci_lower", "ci_upper"], 
            index=t_grid
        )
        for t in t_grid:
            exog = self._get_exog(pd.Series([t] * len(self.data), index=self.data.index))
            prediction = self.estimator.get_prediction(exog.mean().to_frame().T)
            response_df.loc[t] = (
                prediction.predicted.item(), prediction.se.item(), 
                2*min(stats.norm.cdf(0, prediction.predicted.item(), prediction.se.item()), stats.norm.sf(0, prediction.predicted.item(), prediction.se.item())),
                prediction.predicted.item() - 1.96*prediction.se.item(),
                prediction.predicted.item() + 1.96*prediction.se.item()
            )
        return response_df


class LinearOLSPoly(LinearOLS):
    description = "Linear OLS with Polynomial Treatment"
    def __init__(self, global_config, exp_config, controls, **kwargs):
        super().__init__(global_config, exp_config, controls, **kwargs)

    def _treatment_featurizer(self, t_series):
        df_t = pd.DataFrame(index=self.data.index)
        df_t["t"] = t_series
        df_t["t2"] = t_series ** 2
        self.treatment_features = df_t.columns.to_list()
        return df_t

    def fit(self, data): 
        super().fit(data)
    