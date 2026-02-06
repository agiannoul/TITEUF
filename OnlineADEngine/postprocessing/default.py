import pandas as pd

from OnlineADEngine.postprocessing.post_processor import PostProcessorInterface


class DefaultPostProcessor(PostProcessorInterface):
    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame) -> list[float]:
        return scores

    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame,anomaly_ranges=None) -> None:
        pass

    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        return score_point
    

    def get_params(self):
        return {}


    def __str__(self) -> str:
        return 'Default'