import abc

import pandas as pd

from OnlineADEngine.pdm_evaluation_types.types import EventPreferences


class PostProcessorInterface(abc.ABC):
    def __init__(self, event_preferences: EventPreferences):
        self.event_preferences = event_preferences

    @abc.abstractmethod
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame, anomaly_ranges=None) -> None:
        """
        Fit the post-processor on the historic data.
        """
        pass

    @abc.abstractmethod
    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame) -> list[float]:
        pass


    @abc.abstractmethod
    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        pass


    @abc.abstractmethod
    def get_params(self):
        pass
    

    @abc.abstractmethod
    def __str__(self) -> str:
        pass