import pandas as pd

from OnlineADEngine.thresholding.thresholder import ThresholderInterface
from OnlineADEngine.pdm_evaluation_types.types import EventPreferences


class ConstantThresholder(ThresholderInterface):
    def __init__(self, event_preferences: EventPreferences, threshold_value: float = 0.5):
        super().__init__(event_preferences=event_preferences)
        self.threshold_value = threshold_value

    def fit(self, historic_data: list, historic_sources: list[str], event_data: pd.DataFrame,anomaly_ranges=None) -> None:
        pass

    def infer_threshold(self, scores: list[float], source: str, event_data: pd.DataFrame, scores_dates: list[pd.Timestamp]) -> list[float]:
        return [self.threshold_value for i in range(len(scores))]
    

    def infer_threshold_one(self, score: float, source: str, event_data: pd.DataFrame) -> float:
        return self.threshold_value


    def get_params(self):
        return {
            'threshold_value': self.threshold_value
        }
    

    def __str__(self) -> str:
        return 'ConstantThresholder'