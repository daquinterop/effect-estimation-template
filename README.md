# Causal Effect Estimation Template

This repository provides a modular boilerplate for estimating causal effects under the unconfoundedness assumption. It is designed to separate data engineering, statistical estimation, and robustness checks into distinct, extendable components.

## Architecture

The framework is built around three core abstract base classes:

- `BaseData`: Handles data transformations and splits. It requires explicitly defining `_transform_outcome`, `_transform_treatment`, and `_transform_controls`. Data is retrieved via a single `get_data()` method.
- `BaseEstimator`: Enforces a statsmodels-style instantiation where data (`outcome`, `treatment`, `controls`) is passed directly to the constructor. The `estimate_ate()` method returns a `CausalEstimate` dataclass containing the effect, standard errors, p-values, and a reference to the estimator instance.
- `BaseRobustnessCheck`: Consumes a `CausalEstimate` object to run refutation tests (e.g., placebo treatments) by reusing the stored estimator and data.

## Usage Example

To use this template in a project, subclass the base components and implement the required abstract methods. 

### 1. Define the Data Module

```python
import pandas as pd
from effect_estimation.data.base import BaseData

class ProjectData(BaseData):
    def _transform_outcome(self):
        # Example: Log transformation
        return np.log(self.df[self.outcome_col])
        
    def _transform_treatment(self):
        # Example: Binarize treatment
        return (self.df[self.treatment_col] > 0).astype(int)
        
    def _transform_controls(self):
        # Example: Select specific confounders
        return self.df[self.control_cols]
```

### 2. Define the Estimator

```python
from effect_estimation.estimators.base import BaseEstimator, CausalEstimate
import statsmodels.api as sm

class OLSEstimator(BaseEstimator):
    def estimate_ate(self) -> CausalEstimate:
        X = sm.add_constant(pd.concat([self.treatment, self.controls], axis=1))
        model = sm.OLS(self.outcome, X).fit()
        
        treatment_name = self.treatment.name
        
        return CausalEstimate(
            value=model.params[treatment_name],
            std_error=model.bse[treatment_name],
            p_value=model.pvalues[treatment_name],
            estimator_instance=self
        )
```

### 3. Run the Pipeline

```python
# Initialize data
df = pd.read_parquet("data.parquet")
data = ProjectData(df, outcome_col="y", treatment_col="t", control_cols=["x1", "x2"])
outcome, treatment, controls = data.get_data()

# Define estimators
estimators = {
    'OLS': OLSEstimator(outcome, treatment, controls),
    'OLS_NoControls': OLSEstimator(outcome, treatment, None)
}

# Estimate effects
estimates = {}
for name, estimator in estimators.items():
    estimates[name] = estimator.estimate_ate()
    print(f"{name} ATE: {estimates[name].value:.4f} (p={estimates[name].p_value:.4f})")

# Run robustness checks (assuming PlaceboCheck is implemented)
# robustness_results = {}
# for name, estimate in estimates.items():
#     robustness_results[name] = PlaceboCheck(estimate)
```

## Environment Management

This project uses `uv` for dependency management. To add dependencies, use `uv add <package>`. When using this template as a base for another project, consider using tools like `Cruft` or `Cookiecutter` to manage upstream boilerplate updates, or simply copy the files if strict one-way synchronization is not required.
