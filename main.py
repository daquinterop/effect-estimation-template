from dowhy.causal_identifier import adjustment_set
import tomllib
import argparse
import os
import pandas as pd
import numpy as np
import logging 
import sys
from pathlib import Path
import json

from src.identification.backdoor_adjustment import identify_adjustment_set
from src.data.base import generate_data, DataImputed, Data, BaseData
from src.estimators import get_estimator

logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout
    )
logger = logging.getLogger("causal_pipeline")

def parse_args():
    parser = argparse.ArgumentParser(description="Causal Effect Estimation Pipeline")
    parser.add_argument('--config', type=str, required=True, help="Path to TOML configuration file")
    parser.add_argument('--experiment', type=str, required=True, help="Name of the experiment to run")
    parser.add_argument('--data', action='store_true', help="Run data preprocessing step")
    parser.add_argument('--identify', action='store_true', help="Run identification step")
    parser.add_argument('--estimate', action='store_true', help="Run estimation step")
    parser.add_argument('--plot', action='store_true', help='Generate plots from saved results')
    parser.add_argument('--refute', action='store_true', help="Run refutation(robustness) checks")
    return parser.parse_args()

def load_config(config_path, experiment_name):
    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)

    if 'experiments' not in config_data or experiment_name not in config_data['experiments']:
        raise ValueError(f"Experiment '{experiment_name}' not found in {config_path}")
        
    exp_config = config_data['experiments'][experiment_name]
    exp_config["name"] = experiment_name
    global_config = config_data.get('global', {})
    return config_data, exp_config, global_config


def generate_fertilizer_data(global_config):
    """
    Runs the data preprocessing step. It must take the 'raw' data, and perform all 
    data transformations to create the final dataset used in the analysis.
    """
    data, _ = generate_data(n_samples=1500, random_state=global_config['random_seed'])
    data.to_csv(
        os.path.join(global_config['data_path'], "raw.csv"), 
        index=True
    )

def create_exp_dirs(global_config, experiment_config):
    exp_path = Path(global_config["outputs_path"]) / experiment_config["name"]
    os.makedirs(exp_path / "estimates", exist_ok=True)
    os.makedirs(exp_path / "refutation", exist_ok=True)

def run_data(global_config):
    """
    Runs the data pre-processing pipeline. All is handled in the Data class. 
    """ 
    data = Data(global_config, cache=False)

def run_identify(global_config, exp_config, cache=False):
    """
    Runs the identification step. The loading and saving of files is handled in
    this function. The identification logic is in the identification module
    """
    exp_path = Path(global_config["outputs_path"]) / exp_config["name"]
    adj_set_name = "{0}-ATE-of-{1}-on-{2}-under-{3}.json".format(
        exp_config['adjustment_set'], global_config['treatment_node'],
        global_config['outcome_node'], exp_config['dag']
    )
    adj_set_path = exp_path / adj_set_name
    if not cache:
        valid_sets = identify_adjustment_set(global_config, exp_config)
        with open(adj_set_path, "w") as f:
            json.dump(valid_sets, f, indent=4)
        logger.info(f"Identified adjustment set: {valid_sets}")
        logger.info(f"Saved adjustment set to: {adj_set_path}")
        return valid_sets
    if cache:
        if adj_set_path.exists():
            with open(adj_set_path, "r") as f:
                valid_sets = json.load(f)
            logger.info(f"Loaded adjustment set from: {adj_set_path}")
            return valid_sets
        else:
            raise FileNotFoundError(f"Adjustment set not found at: {adj_set_path}. Run identification step first.")

def run_estimate(global_config, exp_config, data: BaseData, adjustment_sets):
    for controls in adjustment_sets:
        logger.info(f"Running estimation for controls: {controls}")
        for estimator_name in exp_config['estimators']:
            estimator_class = get_estimator(estimator_name)
            estimator = estimator_class(
                global_config, exp_config, controls
            )
            estimator.fit(data)
            # Estimate dose-response curve
            t_grid = np.linspace(*data.treatment_range, num=100)
            dose_response_df = estimator.dose_response_curve(t_grid)
            # dose_response_df = pd.DataFrame(
            #     index=t_grid, 
            #     columns=["estimate", "std_error", "p_value", "ci_lower", "ci_upper"]
            # )
            # for t in t_grid:
            #     estimate = estimator.dose_response_curve(t)
                # dose_response_df.loc[t, "estimate"] = estimate.value
                # dose_response_df.loc[t, "std_error"] = estimate.std_error
                # dose_response_df.loc[t, "p_value"] = estimate.p_value
                # dose_response_df.loc[t, "ci_lower"] = estimate.ci_lower
                # dose_response_df.loc[t, "ci_upper"] = estimate.ci_upper
            
            file_name = f"{estimator_name}_average_dose_response.csv"
            save_dir = Path(global_config["outputs_path"]) / exp_config["name"] / "estimates"
            dose_response_df.to_csv(save_dir / file_name, index=True)
            

def main():
    args = parse_args()
    logger.info(f"Running Experiment: {args.experiment}")
    config_data, exp_config, global_config = load_config(args.config, args.experiment)
    create_exp_dirs(global_config, exp_config)

    if args.data:
        data = run_data(global_config)

    if args.identify:
        run_identify(global_config, exp_config)

    if args.estimate:
        data = Data(global_config, cache=True)
        adjustment_sets = run_identify(global_config, exp_config, cache=True)
        run_estimate(global_config, exp_config, data, adjustment_sets)

if __name__ == "__main__":
    main()
    logger.info("Finished pipeline.")
