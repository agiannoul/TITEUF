import pandas as pd
import xgboost as xgb
from OnlineADEngine.method.supervised_method import SupervisedMethodInterface
from OnlineADEngine.pdm_evaluation_types.types import EventPreferences
import numpy as np

def create_windowed_data(df: pd.DataFrame, seq_len: int):
    feature_cols = [c for c in df.columns if c not in ["vehicle_id", "label"]]

    all_X, all_y = [], []

    for vid, vdf in df.groupby("vehicle_id"):
        vdf = vdf.reset_index(drop=True)
        vdf[feature_cols] = vdf[feature_cols].astype("float32")

        data = vdf[feature_cols].values
        labels = vdf["label"].values.astype("float32")

        padded = np.pad(data, ((seq_len - 1, 0), (0, 0)), mode="edge")

        X = np.stack([padded[i:i + seq_len] for i in range(len(data))])
        
        all_X.append(X.reshape(X.shape[0], -1))
        all_y.append(labels)

    X_final = np.concatenate(all_X)
    y_final = np.concatenate(all_y)
    

    # if np.isnan(y_final.numpy()).any():
    #     print("❌ Inf inside windowed X")
    # if np.isinf(y_final.numpy()).any():
    #     print("❌ Inf inside windowed X")
    return X_final, y_final


class XGBoostWRUL(SupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences,save_model=False,seq_len=10, *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.model_per_source = {}
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.save_model=save_model
        self.seq_len=seq_len


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
            self.model_per_source[current_historic_source] = xgb.XGBRegressor(*self.initial_args,
                                                                               **self.initial_kwargs)
            
            df = current_historic_data.copy()
            df["label"] = labels
            ds,Y = create_windowed_data(df, self.seq_len)            

            self.model_per_source[current_historic_source].fit(ds, Y)

            if self.save_model:
                import pickle
                with open(f"xgboost_w_model_{current_historic_source}.pkl", "wb") as f:
                    pickle.dump(self.model_per_source[current_historic_source], f)
    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        # TODO need to check if a model is available for the provided source
        
        df = target_data.copy()
        df["label"] = 0
        ds,_ = create_windowed_data(df, self.seq_len)
        predictions= self.model_per_source[source].predict(ds)[:].tolist()
        predictions= [x for x in predictions]
        return predictions

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        return self.model_per_source[source].predict_proba([new_sample.to_numpy()])[:, 1].tolist()[0]

    def get_params(self) -> dict:
        return {
            **(xgb.XGBRegressor(novelty=False, *(self.initial_args), **(self.initial_kwargs)).get_params()),
        }
    def get_library(self) -> str:
        # TODO we could also try to return a reference to the corresponding subpackage if it works
        return 'no_save'

    def __str__(self) -> str:
        """
            Returns a string representation of the corresponding method
        """
        return "XGBOOST_W"
    def get_all_models(self):
        pass