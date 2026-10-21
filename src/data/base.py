from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Any, List
from pathlib import Path
import json

import warnings
import logging

logger = logging.getLogger("causal_pipeline")

class BaseData(ABC):
    """
    Abstract Base Class for Causal Data Modules.
    
    This is the base class for the data. The idea is that all the data operations. The data
    operations such as cleaning, transformation or imputation are implemented as methods of 
    this class. 

    The child of this class represents a given data used for analysis. That entails a single
    raw data source, and the given pre-processing steps. This allows to define multiple data 
    pre-processing pipelines by only changing specific pre-processing steps in the child class.

    The splits of the data, used for cross-fitting methods, are defined in this data class.
    This is a design choice as how the data is split deppends mostly on characteristics of the
    data and not so much on the estimation method.
    """
    
    def __init__(self, global_config: dict, cache: bool = False):
        """
        Initialize the Data object.
        
        Args:
            global_config: global configuration dictionary.
            processed_name: Name of the processed data file. The first column must be the index.
            cache: Whether to load the processed data from file or generate it from raw data.
        """

        self._raw_data = pd.read_csv(Path(global_config['data_path']) / global_config['raw_data_file'], index_col=0)
        self.outcome_col = global_config['outcome_node']
        self.treatment_col = global_config['treatment_node']
        self._data_ranges_file = global_config.get('data_ranges', None)
        self._processed_file = f"processed__{self.__class__.__name__}.csv"

        if cache:
            try:
                self.data = pd.read_csv(Path(global_config['data_path']) / self._processed_file, index_col=0)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"No processed data found at {Path(global_config['data_path']) / self._processed_file}. "
                    "Run the data preprocessing step by adding the `--data` flag."
                )
        else:
            self.data = self._raw_data.copy()
            self._clean_data()
            self._impute_missing()
            self.data = self.data.dropna(how="any", axis=0)
            self.data.to_csv(Path(global_config['data_path']) / self._processed_file, index=True)
            logger.info(f"Data saved to {Path(global_config['data_path']) / self._processed_file}")

    def _transform_data(self) -> Any:
        """
        Transform the variables as needed.
        """
        pass

    @abstractmethod
    def _clean_data(self) -> Any:
        """
        Filters the data to remove any invalid entries. E.g. remove rows with NaN values, 
        remove rows where treatment is outside of desired range. A basic cleaning is implemented
        on which the data is filtered based on the 'data_ranges' parameter defined in the 
        configuration file. 
        """
        if self._data_ranges_file is None:
            logger.warning("No data ranges defined. Skipping data cleaning.")
            return

        with open(self._data_ranges_file, 'rb') as f:
            data_ranges = json.load(f)

        for col, (min_val, max_val) in data_ranges.items():
            self.data[col] = self.data[col].where(
                lambda x: (x >= min_val) & (x <= max_val),
                np.nan
            )
        logger.info(f"Data cleaning pipeline ran")

    @abstractmethod
    def _impute_missing(self) -> Any:
        """
        Imputes missing values in the data.
        """
        pass

    def get_splits(self) -> Any:
        """
        Defines how splits are made for cross-fitting methods (e.g., DML).
        
        Returns:
            An iterable yielding (train, test) splits as arrays of indices
            based on user-defined logic (groups, strata, etc).
        """
        pass

def true_average_dose_response(t_eval, a, b, c):
    """
    Calculates the Marginal Average Dose-Response analytically.
    """
    y_t = np.mean(a * (1 - np.exp(-c * (t_eval + b))))
    return y_t
    

class Data(BaseData):
    """
    Basic data with no imputation of missing values.
    """
    def __init__(self, global_config, cache: bool = False):
        super().__init__(
            global_config, cache = cache
        )

    def _clean_data(self):
        super()._clean_data()
        
    def _impute_missing(self):
        pass

    def get_splits(self):
        pass


class DataImputed(Data):
    """
    Data where missing values are imputed using the mean of the variable.
    """
    def _impute_missing(self):
        self.data = self.data.fillna(self.data.mean())
        logger.info(f"Data imputation pipeline ran")


def generate_data(n_samples=1500, random_state=42):
    """
    Generates synthetic fertilizer and yield data following a Mitscherlich dose-response.
    """
    np.random.seed(random_state)
    
    # Confounders (X)
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = np.random.normal(0, 1, n_samples)
    x4 = np.random.normal(0, 1, n_samples)
    x5 = np.random.normal(0, 1, n_samples)
    
    X = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})
    
    # Treatment T (N-rate)
    # x1, x2, x3 increase fertilizer usage Propensity
    mu_T = 100 + 30 * x1 + 30 * x5 + 30 * x3 
    mu_T = np.clip(mu_T, 10, 250)
    T = np.random.normal(mu_T, 20)
    T = np.clip(T, 0, 250)
    
    # Outcome parameters
    # a: Maximum yield. x1 increases 'a', x2 decreases 'a', x3 has no effect.
    a = 5000 + 400 * x1 - 800 * x2 + 0 * x3 + 200 * x4
    a = np.clip(a, 1000, 8000)  # Ensure maximum yield is strictly positive
    # a = 4000*np.ones_like(x1)
    
    # c: curvature
    c = 0.015 + x5 *0.006

    # b: soil inherent nutrient, y0: baseline fertility
    y0 = 2000 + 500 * x5
    y0 = np.clip(y0, 250, 5000)
    y0 = np.where(a > y0, y0, 0.3*a)
    # y0 = 2000*np.ones_like(x5)
    b = -1/c*np.log(1-y0/a)
       
    # True Outcome Y
    Y = a * (1 - np.exp(-c * (T + b))) + np.random.normal(0, 200, n_samples)
    Y = np.maximum(Y, 0)  # Ensure yield is never negative

    df = pd.DataFrame({
        'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5,
        't': T, 'y': Y
    })
    return df, (a, b, c)