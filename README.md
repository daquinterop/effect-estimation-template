# Causal Effect Estimation Template

This repository provides a modular boilerplate for estimating causal effects under the unconfoundedness assumption. It separates data engineering, DAG-based causal identification, statistical estimation (dose-response curves and treatment efficiencies), and robustness checks into distinct, extendable components.

As an example, the template includes a synthetic dataset generated with crop models where the true dose-response relationship is known, allowing you to benchmark and compare estimator performance against ground truth under different causal DAG assumptions.

## Architecture

The framework is organized into modular components:

- **Data Engineering (`src.data.base.BaseData`)**: Handles data loading, transformation, missing value imputation, and caching. Subclasses implement `_load_data()`, `_transform_data()`, and `_impute_missing()`.
- **Causal Identification (`src.identification.backdoor_adjustment`)**: Reads DAG specifications (`config/<dag>.json`) and uses the DoWhy backdoor criterion to identify valid adjustment sets.
- **Statistical Estimators (`src.estimators.base.BaseEstimator`)**: Defines a standardized interface for causal estimators. The causal estimands of interest (e.g. ATE, ATT, CATE, Dose-Response, etc.) are defined as methods of the estimators. This repository implements an example of fertilizer-dose-response estimation (Quadratic OLS, Generalized Propensity Score, and Random Forest S-Learner) evaluated against known ground-truth responses.
- **Robustness & Diagnostics (`src.robustness.base.BaseRobustnessCheck`)**: Evaluates refutation tests and sensitivity checks using `CausalEstimate` outputs. It has not been implemented yet. 

---

## Configuration & Multi-Experiment Specifications

The pipeline is entirely configuration-driven via [`config/experiments.toml`](config/experiments.toml). This allows you to define multiple experiments side-by-side and systematically evaluate alternative specifications (e.g., different identification criteria such as DAGs or adjustment sets, or varying sets of estimators) without modifying pipeline code.

### Configuration Structure

```toml
[global]
treatment_node = "treatment"
outcome_node = "outcome"
conditional_nodes = ["modifier"]
random_seed = 43
data_path = "data/"
raw_data_file = "raw.csv"
outputs_path = "outputs/"
data_ranges = "config/data_ranges.json"

# Experiment Specification 1: Complete DAG with valid confounders
[experiments.real]
dag = "dag-real"
adjustment_set = "BACKDOOR_EFFICIENT"
estimators = ["LinearOLSPoly", "GPS", "SLearner"]

# Experiment Specification 2: Alternative DAG
[experiments.alternative_spec]
dag = "dag-alternative"
adjustment_set = "BACKDOOR_EFFICIENT"
estimators = ["LinearOLSPoly", "GPS", "SLearner"]
```

### Key Configuration Elements

1. **`[global]`**:
   - `treatment_node` & `outcome_node`: The causal treatment and target outcome variable names.
   - `conditional_nodes`: List of variables for conditional effect estimation (if any).
   - `random_seed`: Seed ensuring reproducibility across bootstrap iterations and stochastic models.
   - `data_path` & `raw_data_file`: Directory and file name for the input data.
   - `outputs_path`: Base directory for artifacts (`outputs/<experiment_name>/`).
   - `data_ranges`: Path to JSON file defining reasonable ranges for data validation.

2. **`[experiments.<experiment_name>]` (Defining Alternative Specifications)**:
   - **Alternative DAG Topologies (`dag`)**: References a JSON DAG structure in `config/<dag>.json`. This enables testing sensitivity to different causal structural assumptions.
   - **Adjustment Criterion (`adjustment_set`)**: Selects the DoWhy backdoor criterion to apply (e.g., `BACKDOOR_EFFICIENT`, `BACKDOOR_MINIMAL`, `BACKDOOR_DEFAULT`).
   - **Estimator Selection (`estimators`)**: Specifies which estimator classes to run (e.g., `["LinearOLSPoly", "GPS", "SLearner"]`). Different experiments can compare different modeling choices or subsets of estimators.

Each experiment produces isolated outputs in `outputs/<experiment_name>/`.
---

## Confounder & DAG Node Naming Convention

> [!IMPORTANT]
> **Column Prefix Convention**: Any column in the raw dataset (`raw.csv`) that represents or belongs to a DAG confounder node **must start with that node's name** followed by an underscore or specific identifier.
>
> For example, if the DAG specifies `weather` and `soil` as confounder nodes:
> - Valid columns for `weather`: `weather_rain_seasonal`, `weather_temp_avg`
> - Valid columns for `soil`: `soil_clay_pct`, `soil_som_pct`
>
> The estimation pipeline dynamically maps identified DAG adjustment nodes to DataFrame features by matching columns that begin with the node name (`col.startswith(node)`). If columns do not adhere to this prefix rule, the estimator will find no control variables for that node.

---

## Running the Pipeline

The project is driven by TOML configurations (e.g. `config/experiments.toml`) and executed through `main.py`.

### Execute an Experiment

Run data preprocessing, causal identification, and estimation:

```bash
# Run the complete DAG experiment (creates processed data on first run)
uv run python main.py --config config/experiments.toml --experiment real --data --identify --estimate

# Run an alternative experiment
uv run python main.py --config config/experiments.toml --experiment alternative_spec --identify --estimate
```

## Environment & Dependency Management

This project uses `uv` for environment and package management:

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package_name>
```