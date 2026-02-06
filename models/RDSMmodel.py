import numpy as np
import pandas as pd
from OnlineADEngine.method.supervised_method import SupervisedMethodInterface
from OnlineADEngine.pdm_evaluation_types.types import EventPreferences
from dsm import DeepRecurrentSurvivalMachines
from sklearn.preprocessing import StandardScaler


class RDSM(SupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences, save_model=False, learning_rate=1e-4, k=3, layers=3,
                 hidden=100, batch_size=32, typ="LSTM", iters=100,to_scale=True, *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.model_per_source = {}
        self.avail_times_per_source = {}
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.save_model = save_model
        self.k = k
        self.layers = layers
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.typ = typ
        self.to_scale = to_scale
        self.iters = iters

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
            # add noise
            self.scaler=StandardScaler()

            num_cols=current_historic_data.columns.difference(['vehicle_id'])
            f_num_cols=[]
            for c in num_cols:
                if current_historic_data[c].unique().shape[0]>2:
                    f_num_cols.append(c)
            self.s_columns=f_num_cols
            df_with_RUL = current_historic_data.copy()
            df_with_RUL[num_cols]=df_with_RUL[num_cols].astype("float64")
            if self.to_scale:
                df_with_RUL[f_num_cols] = self.scaler.fit_transform(df_with_RUL[f_num_cols])

            df_with_RUL["RUL"] = [lb[0] if lb[0]>0 else 0.01 for lb in labels]
            df_with_RUL["event"] = [lb[1] for lb in labels]
            X_seq = []
            t_seq = []
            e_seq = []
            for vechicle, groupdf in df_with_RUL.groupby("vehicle_id"):
                t_seq.append([r for r in groupdf["RUL"]])
                e_seq.append([r for r in groupdf["event"]])
                X_seq.append(groupdf.drop(["RUL", "event", "vehicle_id"], axis=1).values)

            x_seq = np.empty(len(X_seq), dtype=object)
            x_seq[:] = X_seq
            T_seq = np.empty(len(t_seq), dtype=object)
            T_seq[:] = t_seq
            E_seq = np.empty(len(t_seq), dtype=object)
            E_seq[:] = e_seq
            model = DeepRecurrentSurvivalMachines(
                k=self.k,
                layers=self.layers,
                typ=self.typ,
                hidden=self.hidden,
                # random_seed=42

            )

            model.fit(
                x_seq,
                T_seq,
                E_seq,
                iters=self.iters,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size
            )

            # RandomSurvivalForest(n_estimators=100, min_samples_split=6, min_samples_leaf=5, verbose=1, n_jobs=4)
            self.model_per_source[current_historic_source] = model
            # self.model_per_source[current_historic_source].fit(current_historic_data, y)
            self.avail_times_per_source[current_historic_source] = np.sort(
                np.unique([ty for ty, ev in zip(df_with_RUL["RUL"], df_with_RUL["event"])]))

            if self.save_model:
                import pickle
                with open(f"RDSM_model_{current_historic_source}.pkl", "wb") as f:
                    pickle.dump(self.model_per_source[current_historic_source], f)

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame):
        # TODO need to check if a model is available for the provided source
        data_to_predict = target_data.drop(["vehicle_id"], axis=1).astype("float64").copy()
        if self.to_scale:
            data_to_predict[self.s_columns] = self.scaler.transform(data_to_predict[self.s_columns])

        predictions = self.model_per_source[source].predict_survival(
            np.array([data_to_predict.values]),
            [tt for tt in self.avail_times_per_source[source]])

        n, T = predictions.shape

        # Repeat the time array for every curve â†’ shape (n, T)
        times_tiled = np.tile(self.avail_times_per_source[source], (n, 1))

        # Stack into (n, 2, T)
        result = np.stack([predictions, times_tiled], axis=1)
        return result

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        pass

    def get_params(self) -> dict:
        params = {'learning_rate': self.learning_rate, 'k': self.k, 'layers': self.layers,
                  'hidden': self.hidden, 'batch_size': self.batch_size,
                  'typ': self.typ, 'iters': self.iters}

        return params

    def get_library(self) -> str:
        # TODO we could also try to return a reference to the corresponding subpackage if it works
        return 'no_save'

    def __str__(self) -> str:
        """
            Returns a string representation of the corresponding method
        """
        return "RDSM"

    def get_all_models(self):
        pass