import argparse
import json
import logging
import os
from pathlib import Path
import sys
import tomllib

import numpy as np
import pandas as pd

from src.data.base import BaseData, Data
from src.estimators import get_estimator
from src.identification.backdoor_adjustment import identify_adjustment_set

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("causal_pipeline")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the causal effect estimation pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Causal Effect Estimation Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML configuration file")
    parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment to run")
    parser.add_argument("--data", action="store_true", help="Run data preprocessing step")
    parser.add_argument("--identify", action="store_true", help="Run identification step")
    parser.add_argument("--estimate", action="store_true", help="Run estimation step")
    parser.add_argument("--plot", action="store_true", help="Generate plots from saved results")
    parser.add_argument("--refute", action="store_true", help="Run refutation (robustness) checks")
    return parser.parse_args()


def load_config(config_path: str | Path, experiment_name: str) -> tuple[dict, dict, dict]:
    """Load configuration from a TOML file.

    Parameters
    ----------
    config_path : str or Path
        Path to TOML configuration file.
    experiment_name : str
        Name of the experiment to run.

    Returns
    -------
    tuple[dict, dict, dict]
        A tuple of (full_config_data, experiment_config, global_config).

    Raises
    ------
    ValueError
        If the experiment name is not found in the configuration.
    """
    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)

    if "experiments" not in config_data or experiment_name not in config_data["experiments"]:
        raise ValueError(f"Experiment '{experiment_name}' not found in {config_path}")

    exp_config = config_data["experiments"][experiment_name]
    exp_config["name"] = experiment_name
    global_config = config_data.get("global", {})
    return config_data, exp_config, global_config


def create_exp_dirs(global_config: dict, experiment_config: dict) -> None:
    """Create directory structure for experiment outputs.

    Parameters
    ----------
    global_config : dict
        Global pipeline configuration dictionary.
    experiment_config : dict
        Experiment-specific configuration dictionary.
    """
    exp_path = Path(global_config["outputs_path"]) / experiment_config["name"]
    os.makedirs(exp_path / "estimates", exist_ok=True)
    os.makedirs(exp_path / "refutation", exist_ok=True)


def run_data(global_config: dict) -> None:
    """Run the data pre-processing step and save processed data.

    Parameters
    ----------
    global_config : dict
        Global pipeline configuration dictionary.
    """
    Data(global_config, cache=False)


def run_identify(global_config: dict, exp_config: dict, cache: bool = False) -> list[list[str]]:
    """Run or load the causal identification step using backdoor adjustment.

    Parameters
    ----------
    global_config : dict
        Global pipeline configuration dictionary.
    exp_config : dict
        Experiment-specific configuration dictionary.
    cache : bool, default=False
        Whether to load a previously saved adjustment set from disk.

    Returns
    -------
    list[list[str]]
        List of identified valid adjustment sets (lists of node names).

    Raises
    ------
    FileNotFoundError
        If cache is True but the cached adjustment set file does not exist.
    """
    exp_path = Path(global_config["outputs_path"]) / exp_config["name"]
    adj_set_name = "{0}-ATE-of-{1}-on-{2}-under-{3}.json".format(
        exp_config["adjustment_set"],
        global_config["treatment_node"],
        global_config["outcome_node"],
        exp_config["dag"],
    )
    adj_set_path = exp_path / adj_set_name

    if not cache:
        valid_sets = identify_adjustment_set(global_config, exp_config)
        with open(adj_set_path, "w") as f:
            json.dump(valid_sets, f, indent=4)
        logger.info(f"Identified adjustment set: {valid_sets}")
        logger.info(f"Saved adjustment set to: {adj_set_path}")
        return valid_sets

    if adj_set_path.exists():
        with open(adj_set_path, "r") as f:
            valid_sets = json.load(f)
        logger.info(f"Loaded adjustment set from: {adj_set_path}")
        return valid_sets

    raise FileNotFoundError(
        f"Adjustment set not found at: {adj_set_path}. Run identification step first."
    )


def run_estimate(
    global_config: dict,
    exp_config: dict,
    data: BaseData,
    adjustment_sets: list[list[str]],
) -> None:
    """Run causal effect estimation across adjustment sets and estimators.

    Parameters
    ----------
    global_config : dict
        Global pipeline configuration dictionary.
    exp_config : dict
        Experiment-specific configuration dictionary.
    data : BaseData
        Preprocessed dataset instance.
    adjustment_sets : list[list[str]]
        List of identified adjustment sets (lists of node names).
    """
    eff_df = pd.DataFrame(
        columns=["value", "std_error", "p_value", "ci_lower", "ci_upper"],
        index=pd.MultiIndex.from_product(
            [range(len(adjustment_sets)), exp_config["estimators"]],
            names=["adjustment_set_n", "estimator"],
        ),
    )
    for set_n, adj_set in enumerate(adjustment_sets):
        logger.info(f"Running estimation for controls: {adj_set}")
        controls = []
        for node in adj_set:
            for col in data.data.columns:
                if col.startswith(node):
                    controls.append(col)

        for estimator_name in exp_config["estimators"]:
            estimator_class = get_estimator(estimator_name)
            estimator = estimator_class(global_config, exp_config, controls)
            estimator.fit(data)

            # Estimate dose-response curve
            t_grid = np.linspace(0, data.treatment_range[1], num=100)
            dose_response_df = estimator.dose_response_curve(t_grid)
            file_name = f"{estimator_name}_average_dose_response_{set_n}.csv"
            save_dir = Path(global_config["outputs_path"]) / exp_config["name"] / "estimates"
            dose_response_df.to_csv(save_dir / file_name, index=True)

            # Estimate treatment efficiency
            treatment_efficiency = estimator.average_treatment_efficiency()
            eff_df.loc[(set_n, estimator_name), "value"] = treatment_efficiency.value
            eff_df.loc[(set_n, estimator_name), "std_error"] = treatment_efficiency.std_error
            eff_df.loc[(set_n, estimator_name), "ci_lower"] = treatment_efficiency.ci_lower
            eff_df.loc[(set_n, estimator_name), "ci_upper"] = treatment_efficiency.ci_upper
            eff_df.loc[(set_n, estimator_name), "p_value"] = treatment_efficiency.p_value

    file_name = "efficiency_estimates.csv"
    save_dir = Path(global_config["outputs_path"]) / exp_config["name"] / "estimates"
    eff_df.to_csv(save_dir / file_name, index=True)


def main() -> None:
    """Execute the causal effect estimation pipeline from CLI arguments."""
    args = parse_args()
    logger.info(f"Running Experiment: {args.experiment}")
    _, exp_config, global_config = load_config(args.config, args.experiment)
    create_exp_dirs(global_config, exp_config)

    if args.data:
        run_data(global_config)

    if args.identify:
        run_identify(global_config, exp_config)

    if args.estimate:
        data = Data(global_config, cache=True)
        adjustment_sets = run_identify(global_config, exp_config, cache=True)
        run_estimate(global_config, exp_config, data, adjustment_sets)


if __name__ == "__main__":
    main()
    logger.info("Finished pipeline.")
