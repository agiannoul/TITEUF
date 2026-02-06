import numpy as np
import pandas as pd
from OnlineADEngine.method.supervised_method import SupervisedMethodInterface
from OnlineADEngine.pdm_evaluation_types.types import EventPreferences
import tensorflow as tf
from sktime.utils import mlflow_sktime

from models.sktime_models.FCN import FCNRegressorWithLR
from models.sktime_models.CNNT import CNNRegressorWithLR
from models.sktime_models.CNTC import CNTCRegressorWithLR
from models.sktime_models.ResNet import ResNetRegressorWithLR
from models.sktime_models.LSTMFCN import LSTMFCNRegressorWithLR
from models.sktime_models.inceptionTime import InceptionTimeRegressorWithLR

from tensorflow.keras.optimizers import Adam
from sktime.regression.kernel_based import RocketRegressor


def normalize_columns(A):
    """
    Normalize a 2D numpy array column-wise.
    If a column is constant (std = 0), the output is zeros for that column.
    """
    A = np.asarray(A, dtype=float)

    mean = A.mean(axis=0)
    std = A.std(axis=0)

    # Identify constant columns
    constant = (std == 0)

    # Avoid division by zero
    std_safe = std.copy()
    std_safe[constant] = 1.0   # dummy value (won't be used)

    # Normalize
    A_norm = (A - mean) / std_safe

    # Set constant columns to zero
    A_norm[:, constant] = 0.0

    return A_norm

def create_windowed_data(df: pd.DataFrame, seq_len: int, normalize: bool = True):
    feature_cols = [c for c in df.columns if c not in ["vehicle_id", "label"]]

    all_X, all_y = [], []

    for vid, vdf in df.groupby("vehicle_id"):
        vdf = vdf.reset_index(drop=True)
        vdf[feature_cols] = vdf[feature_cols].astype("float32")

        data = vdf[feature_cols].values
        labels = vdf["label"].values.astype("float32")

        padded = np.pad(data, ((seq_len - 1, 0), (0, 0)), mode="edge")

        X = np.stack([padded[i:i + seq_len] for i in range(len(data))])
        # normalize

        if normalize:
            X=normalize_columns(X)
            print("NORM")
        else:
            print("NO NORM")
        all_X.append(X)
        all_y.append(labels)

    X_final = np.concatenate(all_X)
    y_final = np.concatenate(all_y)

    # if np.isnan(y_final.numpy()).any():
    #     print("❌ Inf inside windowed X")
    # if np.isinf(y_final.numpy()).any():
    #     print("❌ Inf inside windowed X")
    return X_final, y_final



class BaseSktimeTSRegressor(SupervisedMethodInterface):
    """Base class for sktime time-series regressors following the RUL interface."""

    MODEL_CLASS = "sktime"  # to be overwritten

    def __init__(self, event_preferences: EventPreferences, seq_length: int,
                 save_model=False, normalize=True,*args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.seq_length = seq_length
        self.save_model = save_model
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.model_per_source = {}
        self.normalize = normalize
    def fit(self,
            historic_data,
            historic_sources,
            event_data,
            anomaly_ranges):
        print("fitting...")

        for df, source, labels in zip(historic_data, historic_sources, anomaly_ranges):
            # import tensorflow as tf
            # print("Eager execution:", tf.executing_eagerly())
            # tf.compat.v1.enable_eager_execution()
            model = self.MODEL_CLASS(*self.initial_args, **self.initial_kwargs)

            # ---- NEW: convert DataFrame → numpy3D windows ----
            df = df.copy()
            df["label"] = labels
            ds, Y = create_windowed_data(df, self.seq_length,self.normalize)
            # X =
            print(ds.shape)
            print(ds[-1,-1,:5])
            print(ds[-2,-1,:5])
            print(ds[-3,-1,:5])
            import tensorflow as tf
            print("Eager execution:", tf.executing_eagerly())
            # tf.compat.v1.enable_eager_execution()
            model.fit(ds, Y)
            self.model_per_source[source] = model

            # optional serialization
            if self.save_model:
                model_path = f"{self.MODEL_CLASS.__name__}_{source}"
                model.save(model_path)

        print("Done fitting.")
    def predict(self, target_data, source, event_data=None):
        if source not in self.model_per_source:
            raise ValueError(f"No model for source '{source}'")
        df = target_data.copy()
        df["label"] = 0
        ds, _ = create_windowed_data(df, self.seq_length,self.normalize)
        preds = self.model_per_source[source].predict(ds)
        return preds.tolist()

    # ------------------------------------------------------------------
    # PREDICT ONE
    # ------------------------------------------------------------------
    def predict_one(self, new_sample, source, is_event):
        pass

    def get_params(self):
        model = self.MODEL_CLASS(*self.initial_args, **self.initial_kwargs)
        to_return = model.get_params()
        to_return['seq_length'] = self.seq_length
        return to_return

    def get_library(self):
        return "no_save"

    def get_all_models(self):
        pass


    def __str__(self):
        return self.MODEL_CLASS.__name__




class CNNTRegressorRUL(BaseSktimeTSRegressor):
    MODEL_CLASS = CNNRegressorWithLR


class CNTCRegressorRUL(BaseSktimeTSRegressor):
    MODEL_CLASS = CNTCRegressorWithLR


class FCNRegressorRUL(BaseSktimeTSRegressor):
    MODEL_CLASS = FCNRegressorWithLR


class InceptionTimeRegressorRUL(BaseSktimeTSRegressor):
    MODEL_CLASS = InceptionTimeRegressorWithLR


class LSTMFCNRegressorRUL(BaseSktimeTSRegressor):
    MODEL_CLASS = LSTMFCNRegressorWithLR


class RocketRegressorRUL(BaseSktimeTSRegressor):
    MODEL_CLASS = RocketRegressor


class ResNetRegressorRUL(BaseSktimeTSRegressor):
    MODEL_CLASS = ResNetRegressorWithLR

def get_extended_sktime_ts_regressor_configs():
    """
    Returns list of tuples:
    (class, name, param_search_space_list)
    and each X will be numpy3D time-series input.
    """
    configs = []

    # Rocket




    # # FCN
    # configs.append((
    #     FCNRegressorRUL,
    #     "sktimeFCN",
    #     [{"n_epochs": [30, 50],
    #       # [{"n_epochs": [5],
    #       "batch_size": [64],
    #       "random_state": [42], "seq_length": [5, 10],
    #       "verbose": [True], "optimizer": ["adam"],
    #       "learning_rate": [1e-5, 5e-5, 1e-4],
    #       "activation": ["relu"],
    #       }]
    # ))
    #
    # # ResNet
    # configs.append((
    #     ResNetRegressorRUL,
    #     "sktimeResNet",
    #     [{"n_epochs": [20, 50], "batch_size": [64, 128],
    #       "random_state": [42],
    #       "verbose": [True], "optimizer": ["adam"],
    #       "learning_rate": [1e-5, 5e-5, 1e-4], "seq_length": [5, 10, 20]}]
    # ))

    # configs.append((
    #     RocketRegressorRUL,
    #     "sktimeRocket",
    #     [{"num_kernels": [1000 ], "random_state": [42], "max_dilations_per_kernel": [32],
    #       "n_features_per_kernel": [4], "seq_length": [1, 10, 20],"n_jobs":[8]}]
    # ))

    # CNN
    # configs.append((
    #     CNNTRegressorRUL,
    #     "sktimeCNN",
    #     [{"kernel_size": [5, 7], "n_epochs": [20, 50, 60], "n_conv_layers": [2],
    #       "batch_size": [64, 128], "avg_pool_size": [3], "random_state": [42]
    #       ,"optimizer": ["adam"],
    #       "learning_rate": [1e-5, 5e-5, 1e-4],
    #       "loss": ["mean_absolute_error"],
    #       "seq_length": [5, 10, 20],
    #       "verbose": [True],
    #       }]
    # ))

    # # CNTC
    # configs.append((
    #     CNTCRegressorRUL,
    #     "sktimeCNTC",
    #     [{"n_epochs": [20, 50], "filter_sizes": [(16, 8)], "kernel_sizes": [(1, 1)],
    #       "rnn_size": [32], "lstm_size": [4], "dense_size": [32], "batch_size": [32],
    #       "random_state": [42], "seq_length": [5, 10, 20],
    #       "verbose": [True],
    #       "learning_rate": [1e-5, 5e-5, 1e-4],
    #       }]
    # ))

    # InceptionTime
    # configs.append((
    #     InceptionTimeRegressorRUL,
    #     "sktimeInceptionTime",
    #     [{"n_epochs": [50, 100], "batch_size": [64],
    #       "random_state": [42], "kernel_size": [40],
    #       "n_filters": [32], "use_residual": [True],
    #       "optimizer": ["adam"], "learning_rate": [1e-5, 5e-5, 1e-4],
    #       "use_bottleneck": [True], "bottleneck_size": [32], "depth": [6], "seq_length": [5, 10, 20],
    #       "verbose": [True], }]
    # ))

    # LSTM-FCN
    # configs.append((
    #     LSTMFCNRegressorRUL,
    #     "sktimeLSTMFCN",
    #     [{"n_epochs": [50, 100], "batch_size": [64, 128],
    #       "random_state": [42], "dropout": [0.8], "kernel_sizes": [(8, 5, 3)],
    #       "filter_sizes": [(128, 256, 128)], "lstm_size": [8], "seq_length": [5, 10, 20],
    #       "optimizer": ["adam"], "learning_rate": [1e-5, 5e-5, 1e-4],
    #       "verbose": [True], }]
    # ))

    return configs
