import random

from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import pandas as pd
import numpy as np

from OnlineADEngine.method.supervised_method import SupervisedMethodInterface
from OnlineADEngine.pdm_evaluation_types.types import EventPreferences


class GradientBoostingSurvival(SupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences, u_sample_rate=1,save_model=False, *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.model_per_source = {}
        self.avail_times_per_source = {}
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.save_model = save_model
        self.sample_rate = u_sample_rate

    def undersample_by_rul(self,ydf, target_df, sample_rate=0.3):
        random.seed(42)

        # Store selected indices
        selected_indices = []
        original_ruls=[rr for rr in ydf['RUL'].values]
        # Get all unique RUL values
        unique_ruls = ydf['RUL'].unique()

        for rul in unique_ruls:
            # Find indices for this RUL
            rul_indices = [i for i in range(len(ydf)) if original_ruls[i] == rul]

            # Compute how many to keep (at least 1)
            n_keep = max(1, int(len(rul_indices) * sample_rate))

            # Randomly sample indices
            sampled = random.sample(rul_indices, n_keep)
            selected_indices.extend(sampled)

        # Sort indices to preserve order (optional)
        selected_indices.sort()

        # Subset both ydf and target_df
        ydf_under = ydf.iloc[selected_indices].reset_index(drop=True)
        target_under = target_df.iloc[selected_indices].reset_index(drop=True)

        return ydf_under, target_under
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame,
            anomaly_ranges: list[list]) -> None:
        """
        This method is used to fit a anomaly detection model in supervised way (training), where the data are passed in form
        of Dataframes along with their respected source and labels.

        :param historic_data: a list of Dataframes (used to fit a semi-supervised model). The `historic_data` list parameter elements should be copied if a corresponding method needs to store them for future processing
        :param historic_sources: a list with strings (names) of the different sources
        :param event_data: event data that are produced from the different sources
        :param anomaly_ranges: labels regarding if the data are normal or not. It is a list of lists, where each inner list corresponds to a source and contains the labels for the data in that source.
        :return: None.
        """

        for current_historic_data, current_historic_source, labels in zip(historic_data, historic_sources,
                                                                          anomaly_ranges):
            print(current_historic_data.shape)
            from sksurv.util import Surv
            ydf=pd.DataFrame({'event': [lb[1] for lb in labels],'RUL': [max(1,lb[0]) for lb in labels]})
            if self.sample_rate<1:
                print(current_historic_data.shape)
                ydf,fit_data=self.undersample_by_rul( ydf, current_historic_data, sample_rate=self.sample_rate)
                print(fit_data.shape)
            else:
                fit_data=current_historic_data
            y = Surv.from_dataframe("event", "RUL", ydf)

            # GradientBoostingSurvivalAnalysis with configurable parameters
            self.model_per_source[current_historic_source] = GradientBoostingSurvivalAnalysis(*self.initial_args,**self.initial_kwargs)
            self.model_per_source[current_historic_source].fit(fit_data, y)
            self.avail_times_per_source[current_historic_source]=np.unique([ty for ty in y['RUL']])

            if self.save_model:
                import pickle
                with open(f"gradient_boosting_survival_{current_historic_source}.pkl", "wb") as f:
                    pickle.dump(self.model_per_source[current_historic_source], f)

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame):
        # TODO need to check if a model is available for the provided source
        predictions = self.model_per_source[source].predict_survival_function(target_data, True)
        n, T = predictions.shape

        # Repeat the time array for every curve → shape (n, T)
        times_tiled = np.tile(self.avail_times_per_source[source], (n, 1))

        # Stack into (n, 2, T)
        result = np.stack([predictions, times_tiled], axis=1)
        return result

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        pass

    def get_params(self) -> dict:
        params = {}
        for i, arg in enumerate(self.initial_args):
            params[f"arg{i}"] = arg
        # include keyword args normally
        params.update(self.initial_kwargs)

        return params

    def get_library(self) -> str:
        # TODO we could also try to return a reference to the corresponding subpackage if it works
        return 'no_save'

    def __str__(self) -> str:
        """
            Returns a string representation of the corresponding method
        """
        return "GradientBoostingSurvival"

    def get_all_models(self):
        pass