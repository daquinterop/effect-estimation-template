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
        pass
    
    def estimate_cate(self) -> CausalEstimate:
        raise NotImplementedError
    
    def dose_response_curve(self, t: float) -> CausalEstimate:
        """
        Calculates the dose response curve at a given value of t. This is the conditional expectation 
        of the outcome given the treatment level t, E[Y(t)].
        """
        exog = self._get_exog(pd.Series([t] * len(self.data), index=self.data.index))
        prediction = self.estimator.get_prediction(exog.mean().to_frame().T)
        estimate = CausalEstimate(
            value=prediction.predicted, 
            std_error=prediction.se, 
            p_value=2*min(stats.norm.cdf(0, prediction.predicted, prediction.se), stats.norm.sf(0, prediction.predicted, prediction.se)), 
            ci_lower=prediction.predicted - 1.96*prediction.se, 
            ci_upper=prediction.predicted + 1.96*prediction.se,
        )
        return estimate


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
    