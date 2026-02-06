import numpy as np
import pandas as pd

from OnlineADEngine.thresholding.thresholder import ThresholderInterface
from OnlineADEngine.pdm_evaluation_types.types import EventPreferences


class SurvToRUL(ThresholderInterface):
    """
    This thresholding methods transforms the survival scores into RUL scores
    by training a threshold on validation data.
    If a threshold is given it uses that threshold to transform the scores.
    """
    def __init__(self, event_preferences: EventPreferences, threshold_value=None):
        super().__init__(event_preferences=event_preferences)
        self.threshold_value = threshold_value

    def fit(self, historic_data: list, historic_sources: list[str], event_data: pd.DataFrame,
            anomaly_ranges=None) -> None:
        """
        Score in survival format are lists of tuples (survival_prob, time_to_event).
        """
        if self.threshold_value is None:
            temp_scores = []
            labs = []
            for current_historic_data, current_historic_source, labels in zip(historic_data, historic_sources,
                                                                              anomaly_ranges):
                if labels[0][1]==0:
                    continue
                temp_scores.extend([sc[0] for sc in current_historic_data])
                labs.extend([lab[0] for lab in labels])
            optimed_threshold = self.optimize_threshold(temp_scores, x=historic_data[0][0][1], true_times=labs)
            self.threshold_value = optimed_threshold
    def optimize_threshold(self,curves, x, true_times):
        thetas = np.linspace(0, 1, 501)
        losses = []

        for theta in thetas:
            preds = np.array([self.predicted_time(c, x, theta) for c in curves])
            loss = np.mean(np.abs(preds - true_times))
            losses.append(loss)

        best_theta = thetas[np.argmin(losses)]
        return best_theta

    def predicted_time(self,curve, x,theta):
        idx = np.where(curve <= theta)[0]
        return x[idx[0]] if len(idx) > 0 else x[-1]


    def infer_threshold(self, scores: list, source: str, event_data: pd.DataFrame,
                        scores_dates: list[pd.Timestamp]) -> list[float]:
        in_scores=np.array(scores)
        return [self.predicted_time(in_scores[i,0], in_scores[i,1],self.threshold_value) for i in range(len(scores))]

    def infer_threshold_one(self, score: float, source: str, event_data: pd.DataFrame) -> float:
        return self.threshold_value

    def get_params(self):
        return {
            'threshold_value': self.threshold_value
        }

    def __str__(self) -> str:
        return 'SurvToRUL_threshold'