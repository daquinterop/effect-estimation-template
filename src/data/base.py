from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("causal_pipeline")


class BaseData(ABC):
    """
    Abstract Base Class for Causal Data Modules.

    This is the base class for data handling in the pipeline. Data operations
    such as cleaning, transformation, or imputation are implemented as methods
    of this class.

    Subclasses represent specific datasets used for analysis, wrapping a raw
    data source and a sequence of preprocessing steps.

    Parameters
    ----------
    global_config : dict
        Global configuration dictionary containing keys such as 'data_path',
        'raw_data_file', 'outcome_node', 'treatment_node', and 'data_ranges'.
    cache : bool, default=False
        Whether to load preprocessed data from file or regenerate it from raw data.

    Attributes
    ----------
    outcome_col : str
        Name of the outcome column.
    treatment_col : str
        Name of the treatment column.
    data : pd.DataFrame or None
        Processed dataset.
    treatment_range : tuple of float or None
        (min, max) range of the treatment variable.
    """

    def __init__(self, global_config: dict, cache: bool = False):
        self._raw_data = pd.read_csv(
            Path(global_config["data_path"]) / global_config["raw_data_file"],
            index_col=0,
        )
        self.outcome_col = global_config["outcome_node"]
        self.treatment_col = global_config["treatment_node"]
        self._data_ranges_file = global_config.get("data_ranges", None)
        self._processed_file = f"processed__{self.__class__.__name__}.csv"
        self.data = None
        self.treatment_range = None

        if cache:
            try:
                self.data = pd.read_csv(
                    Path(global_config["data_path"]) / self._processed_file,
                    index_col=0,
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"No processed data found at {Path(global_config['data_path']) / self._processed_file}. "
                    "Run the data preprocessing step by adding the `--data` flag."
                )
        else:
            self.data = self._raw_data.copy()
            self._clean_data()
            self._transform_data()
            self._impute_missing()
            self.data = self.data.dropna(how="any", axis=0)
            self.data.to_csv(
                Path(global_config["data_path"]) / self._processed_file,
                index=True,
            )
            logger.info(
                f"Data saved to {Path(global_config['data_path']) / self._processed_file}"
            )

        self.treatment_range = (
            self.data[self.treatment_col].min(),
            self.data[self.treatment_col].max(),
        )

    def _transform_data(self) -> Any:
        """
        Transform variables as needed (e.g., one-hot encode categorical features).

        Returns
        -------
        None
        """
        categorical_variables = self.data.select_dtypes(
            include=["object", "category"]
        ).columns
        self.data = pd.get_dummies(
            self.data, columns=categorical_variables, drop_first=True, dtype=float
        )

    @abstractmethod
    def _clean_data(self) -> Any:
        """
        Filter data to remove invalid entries based on range constraints.

        Returns
        -------
        None
        """
        if self._data_ranges_file is None:
            logger.warning("No data ranges defined. Skipping data cleaning.")
            return

        with open(self._data_ranges_file, "rb") as f:
            data_ranges = json.load(f)

        for col, (min_val, max_val) in data_ranges.items():
            self.data[col] = self.data[col].where(
                lambda x: (x >= min_val) & (x <= max_val), np.nan
            )
        logger.info("Data cleaning pipeline ran")

    @abstractmethod
    def _impute_missing(self) -> Any:
        """
        Impute missing values in the dataset.

        Returns
        -------
        None
        """
        pass

    def get_splits(self) -> Any:
        """
        Define how data is split for cross-fitting methods (e.g., DML).

        Returns
        -------
        iterable or None
            An iterable yielding (train, test) splits as arrays of indices.
        """
        pass

class Data(BaseData):
    """
    Basic data implementation with no imputation of missing values.

    Parameters
    ----------
    global_config : dict
        Global configuration dictionary.
    cache : bool, default=False
        Whether to load preprocessed data from file or regenerate from raw data.
    """

    def __init__(self, global_config: dict, cache: bool = False):
        super().__init__(global_config, cache=cache)

    def _clean_data(self):
        super()._clean_data()

    def _impute_missing(self):
        pass

    def get_splits(self):
        pass


class DataImputed(Data):
    """
    Data implementation where missing values are imputed with the mean.

    Parameters
    ----------
    global_config : dict
        Global configuration dictionary.
    cache : bool, default=False
        Whether to load preprocessed data from file or regenerate from raw data.
    """

    def _impute_missing(self):
        self.data = self.data.fillna(self.data.mean(numeric_only=True))
        logger.info("Data imputation pipeline ran")