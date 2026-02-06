import pandas as pd
from OnlineADEngine.method.supervised_method import SupervisedMethodInterface
from OnlineADEngine.pdm_evaluation_types.types import EventPreferences
from tabpfn import TabPFNRegressor
import numpy as np
from tabpfn.constants import ModelVersion

class TABPFNRUL(SupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences,save_model=False,n_samples=1000, *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.model_per_source = {}
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.save_model=save_model
        self.n_samples=n_samples
        
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

            self.model_per_source[current_historic_source] = TabPFNRegressor.create_default_for_version(ModelVersion.V2,n_estimators=2)
            n_samples = self.n_samples
            n_rows = len(current_historic_data)

            if n_rows > n_samples:
                # 1. Sample POSITIONS once (fast, no repetition)
                sampled_pos = np.random.RandomState(42).choice(
                    n_rows,
                    size=n_samples,
                    replace=False
                )

                # 2. Slice using iloc for both data and labels
                current_historic_data = current_historic_data.iloc[sampled_pos]
                labels = [labels[i] for i in sampled_pos]
            self.model_per_source[current_historic_source].fit(current_historic_data, labels)
            print("FITTED")
            # if self.save_model:
            #     import pickle
            #     with open(f"xgboost_model_{current_historic_source}.pkl", "wb") as f:
            #         pickle.dump(self.model_per_source[current_historic_source], f)

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        # TODO need to check if a model is available for the provided source
        print(" a ")
        limit=100
        if target_data.shape[0]>limit:
            predictions=[]
            pre=0
            for i in range(0,target_data.shape[0],limit):
                pre=min(i+limit,target_data.shape[0])
                batch_data=target_data[i:min(i+limit,target_data.shape[0])]
                batch_predictions= self.model_per_source[source].predict(batch_data).tolist()
                for i in range(len(batch_predictions)):
                    if np.isnan(batch_predictions[i]):
                        if i==0:
                            batch_predictions[i]=0.0
                        else:
                            batch_predictions[i]=batch_predictions[i-1]
                predictions=predictions+batch_predictions
            if pre<target_data.shape[0]:
                batch_data=target_data[pre:target_data.shape[0]]
                batch_predictions= self.model_per_source[source].predict(batch_data).tolist()
                for i in range(len(batch_predictions)):
                    if np.isnan(batch_predictions[i]):
                        if i==0:
                            batch_predictions[i]=0.0
                        else:
                            batch_predictions[i]=batch_predictions[i-1]
                predictions=predictions+batch_predictions
        else:
            predictions= self.model_per_source[source].predict(target_data)[:].tolist()
        predictions= [x for x in predictions]
        return predictions

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        return self.model_per_source[source].predict_proba([new_sample.to_numpy()])[:, 1].tolist()[0]

    def get_params(self) -> dict:
        return {
           "n_samples": self.n_samples,
        }
    def get_library(self) -> str:
        # TODO we could also try to return a reference to the corresponding subpackage if it works
        return 'no_save'

    def __str__(self) -> str:
        """
            Returns a string representation of the corresponding method
        """
        return "TABPFNv2"
    def get_all_models(self):
        pass