import tomllib
import argparse
import os
import pandas as pd
import logging 
import sys

from src.identification.backdoor_adjustment import identify_adjustment_set
from src.data.base import generate_data, DataImputed, Data


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

def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout
    )
    return logging.getLogger("causal_pipeline")

def load_config(config_path, experiment_name):
    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)

    if 'experiments' not in config_data or experiment_name not in config_data['experiments']:
        raise ValueError(f"Experiment '{experiment_name}' not found in {config_path}")
        
    exp_config = config_data['experiments'][experiment_name]
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

def run_data(global_config):
    data = Data(global_config, cache=False)

def run_identify(config_data, exp_config, global_config):
    dag_path = f"config/{exp_config.get('dag')}"
    dag_basename = os.path.splitext(os.path.basename(exp_config.get('dag')))[0]    

def main():
    args = parse_args()
    config_data, exp_config, global_config = load_config(args.config, args.experiment)

    logger = get_logger()
    logger.info(f"Running Experiment: {args.experiment}")

    if args.data:
        data = run_data(global_config)

    if args.identify:
        run_identify(exp_config, global_config)

    if args.estimate:
        data = Data(global_config, cache=True)

if __name__ == "__main__":
    main()
