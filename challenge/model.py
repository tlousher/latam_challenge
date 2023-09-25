import pandas as pd

from typing import Tuple, Union, List


class DelayModel:

    def __init__(
            self
    ):
        self._model = None  # Model should be saved in this attribute.

    @staticmethod
    def preprocess(
            data: pd.DataFrame,
            target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        if target_column is not None:
            target = data[target_column]
            data = data.drop(columns=[target_column])
            return data, target
        else:
            return data

    def fit(
            self,
            features: pd.DataFrame,
            target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        return

    def predict(
            self,
            features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            raise ValueError("Model has not been trained yet.")
        predictions = self._model.predict(features)
        return predictions
