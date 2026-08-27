from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Optional, Any, List

class BaseData(ABC):
    """
    Abstract Base Class for Causal Data Modules.
    
    This class enforces a structure where data transformations are explicitly
    defined in private methods and executed once via `get_data()`.
    """
    
    def __init__(self, df: pd.DataFrame, outcome_col: str, treatment_col: str, control_cols: List[str]):
        """
        Initialize the Data module.
        
        Args:
            df: The raw pandas DataFrame.
            outcome_col: Name of the outcome variable column.
            treatment_col: Name of the treatment variable column.
            control_cols: List of names of the control/confounder variables.
        """
        self.df = df
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.control_cols = control_cols

    @abstractmethod
    def _transform_outcome(self) -> Any:
        """
        Transform the outcome variable as needed.
        
        Returns:
            Transformed outcome data (e.g. pd.Series or np.ndarray).
        """
        pass

    @abstractmethod
    def _transform_controls(self) -> Any:
        """
        Transform the control variables as needed.
        (e.g., scaling, encoding categoricals, imputing missing values)
        
        Returns:
            Transformed control data (e.g. pd.DataFrame or np.ndarray).
        """
        pass

    @abstractmethod
    def _transform_treatment(self) -> Any:
        """
        Transform the treatment variable as needed.
        (e.g., binarizing, encoding)
        
        Returns:
            Transformed treatment data.
        """
        pass

    def get_data(self) -> Tuple[Any, Any, Any]:
        """
        Runs the transformation pipelines and returns the processed data.
        
        Returns:
            A tuple of (outcome_trans, treatment_trans, controls_trans).
        """
        outcome_trans = self._transform_outcome()
        treatment_trans = self._transform_treatment()
        controls_trans = self._transform_controls()
        return outcome_trans, treatment_trans, controls_trans

    def get_splits(self) -> Any:
        """
        Defines how splits are made for cross-fitting methods (e.g., DML).
        
        Returns:
            Split indices based on user-defined logic (groups, strata, etc).
        """
        pass
