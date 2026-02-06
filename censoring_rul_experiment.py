import argparse
from tkinter.font import names

from models.xgboostRUL import XGBoostRUL
from models.sktime_wrapper import FCNRegressorRUL
from models.sktime_wrapper import ResNetRegressorRUL
from models.sktime_wrapper import LSTMFCNRegressorRUL
from models.sktime_wrapper import CNNTRegressorRUL
from models.sktime_wrapper import InceptionTimeRegressorRUL
from models.sktime_wrapper import RocketRegressorRUL
from models.sklearn_wraper import ElasticNetRUL
from models.sklearn_wraper import RandomForestRUL
from utils import load_HNEI_censored

from models.tabpfnreg import TABPFNRUL
# from models.CatBoost_W import CatBoostWRUL

rul_name_to_class={
    "TABPFNv2": TABPFNRUL,
    "XGBoost":XGBoostRUL,
    # "CatBoost_W_RUL":CatBoostWRUL,
    "RandomForestRUL":RandomForestRUL,
    "ElasticNetRUL":ElasticNetRUL,
    "sktimeRocket":RocketRegressorRUL,
    "sktimeLSTMFCN":LSTMFCNRegressorRUL,
    "sktimeInceptionTime":InceptionTimeRegressorRUL,
    "sktimeCNN":CNNTRegressorRUL,
    "sktimeResNet":ResNetRegressorRUL,
    "sktimeFCN":FCNRegressorRUL,
}


from utils import run_rul_train_val_test


param_space_configurations={
    "TABPFNv2":[{"default": [True],"n_samples":[800,1000]}],
    "XGBoost":[{"n_estimators":[1000,500,200],           # increase, but use early stopping
            "learning_rate":[0.01,0.02,0.001],           # smaller learning rate
            "max_depth":[4,5,6],
            "min_child_weight":[7,10,15],       # shallower trees generalize better
            "random_state":[42],
            "n_jobs":[-1],
            "eval_metric":["rmse"], }],
    "CatBoost_W_RUL": [{"n_estimators":[800,1000,1500,None],           # increase, but use early stopping
            "learning_rate":[0.1,0.05],           # smaller learning rate
            "max_depth":[6,None],
            "random_state":[42],
            "seq_len":[10],
             }],
    "RandomForestRUL": [{"n_estimators": [100, 300],
                                   "max_depth": [5, 10, None],
                                   "n_jobs": [8],
                                   "min_samples_leaf": [1, 5, 10],
                                   "random_state": [42],
                                   "min_samples_split": [2,5,10],
                                   "max_features": ["sqrt", "log2"],
                                   }],
    "ElasticNetRUL": [{"alpha": [0.001, 0.01, 0.1],
                               "l1_ratio": [0.2, 0.5, 0.8],
                               "max_iter": [1000, 2000,3000],
                               "random_state": [42],
                               "method_selection": ["cyclic", "random"]
                               }],
    "sktimeRocket": [{"num_kernels": [1000 ], "random_state": [42],
                                    "max_dilations_per_kernel": [32],
          "n_features_per_kernel": [4], "seq_length": [1, 10, 20],"n_jobs":[8],"normalize": [False, True],}],
    "sktimeLSTMFCN": [{"n_epochs": [50, 100], "batch_size": [64, 128],
          "random_state": [42], "dropout": [0.8], "kernel_sizes": [(8, 5, 3)],
          "filter_sizes": [(128, 256, 128)], "lstm_size": [8], "seq_length": [5, 10, 20],
          "optimizer": ["adam"], "learning_rate": [1e-5, 5e-5, 1e-4,1e-3],
          "verbose": [True],"normalize": [False, True], }],
    "sktimeInceptionTime": [{"n_epochs": [50, 100], "batch_size": [64],
          "random_state": [42], "kernel_size": [40],
          "n_filters": [32], "use_residual": [True],
          "optimizer": ["adam"], "learning_rate": [1e-5, 5e-5, 1e-4,1e-3],
          "use_bottleneck": [True], "bottleneck_size": [32], "depth": [6], "seq_length": [5, 10, 20],
          "verbose": [True],"normalize": [False, True], }],
    "sktimeCNN":[{"kernel_size": [5, 7],
                  "n_epochs": [20, 50,80,100],
                  "n_conv_layers": [2],
          "batch_size": [64, 128], "avg_pool_size": [2],#[3],
                  "random_state": [42]
          ,"optimizer": ["adam"],
          "learning_rate": [1e-5, 5e-5, 1e-4,1e-3,1e-2],
          # "learning_rate": [1e-2],
          "loss": ["mean_absolute_error"],
          "seq_length": [5, 10, 20],
          "verbose": [True],
        "normalize": [False, True],
          }],
    "sktimeResNet":[{"n_epochs": [20, 50,100], "batch_size": [64, 128],
          "random_state": [42],
          "verbose": [True], "optimizer": ["adam"],
          "learning_rate": [1e-5, 5e-5, 1e-4,1e-3], "seq_length": [5, 10, 20],
                     "normalize": [False, True],}],
    "sktimeFCN":[{"n_epochs": [30, 50,100],
          # [{"n_epochs": [5],
          "batch_size": [64],
          "random_state": [42], "seq_length": [5, 10],
          "verbose": [True], "optimizer": ["adam"],
          "learning_rate": [1e-5, 5e-5, 1e-4,1e-3],
          "activation": ["relu"],
          "normalize": [False, True],
          }],
    }



HNEI_rul_best_configurations={
    "TABPFNv2": {'method_default': True, 'method_n_samples': 250},
    "XGBoost":{'method_random_state': 42,'method_n_jobs': -1, 'method_n_estimators': 1000,
             'method_min_child_weight': 10,
              'method_max_depth': 4, 'method_learning_rate': 0.02, 'method_eval_metric': 'mape',},
    "CatBoost_W_RUL": { 'method_learning_rate': 0.05,
                               'method_max_depth': None, 'method_n_estimators': 1500,
                               'method_random_state': 42, 'method_seq_len': 10},
    "RandomForestRUL": {"method_n_estimators": 100,
                                   "method_max_depth":  None,
                                   "method_n_jobs": 8,
                                   "method_min_samples_leaf":1,
                                   "method_random_state": 42,
                                   "method_min_samples_split": 5,
                                   "method_max_features": "sqrt",
                                   },
    "ElasticNetRUL": {"method_alpha":  0.001,
                               "method_l1_ratio":  0.8,
                               "method_max_iter":  2000,
                               "method_random_state": 42,
                               "method_method_selection": "cyclic"
                               },
    "sktimeRocket": {"method_num_kernels": 1000, "method_random_state": 42,"method_max_dilations_per_kernel": 32,
                     "method_n_features_per_kernel":4, "method_seq_length": 10,"method_n_jobs":8,
                     'method_normalize':False},
    "sktimeLSTMFCN":{'method_verbose': True,
              'method_seq_length': 20, 'method_random_state': 42, 'method_optimizer': 'adam',
              'method_n_epochs': 100, 'method_lstm_size': 8, 'method_learning_rate': 0.0015,
              'method_kernel_sizes': (8, 5, 3), 'method_filter_sizes': (128, 256, 128),
              'method_dropout': 0.8, 'method_batch_size': 64, 'init_profile_size': 2,
                     'method_normalize':True},
    "sktimeInceptionTime": {"method_n_epochs": 150, "method_batch_size": 64,
          "method_random_state": 42, "method_kernel_size": 40,
          "method_n_filters": 32, "method_use_residual": True,
          "method_optimizer": "adam", "method_learning_rate": 0.001,
          "method_use_bottleneck": True, "method_bottleneck_size": 32, "method_depth":6,
               "method_seq_length":5,
          "method_verbose": True,'method_normalize':True},
    "sktimeCNN":{"method_kernel_size": 7, "method_n_epochs":  150, "method_n_conv_layers": 2,
          "method_batch_size": 64, "method_avg_pool_size": 2, "method_random_state": 42
          ,"method_optimizer": "adam",
          "method_learning_rate": 0.05,
          "method_loss": "mean_absolute_error",
          "method_seq_length": 20,
          "method_verbose": True,'method_normalize':False
          },
    "sktimeResNet":{'method_verbose': True,
                'method_seq_length': 20, 'method_save_model': False, 'method_random_state': 42,
                'method_optimizer': 'adam', 'method_n_epochs': 60, 'method_learning_rate': 0.0001,
                'method_batch_size': 64, 'init_profile_size': 2,
                    'method_normalize':False},
    "sktimeFCN":{'method_verbose': True,
                'method_seq_length': 10, 'method_save_model': False, 'method_random_state': 42,
                'method_optimizer': 'adam', 'method_n_epochs': 100, 'method_learning_rate': 0.05,
                'method_batch_size': 64, 'method_activation': 'relu', 'init_profile_size': 2,
                 'method_normalize':False},
    }


keep_identifiers_dict={
    "TABPFNv2": False,
    "XGBoost": False,
    "CatBoost_W_RUL": True,
    "RandomForestRUL": False,
    "ElasticNetRUL": False,
    "sktimeRocket": True,
    "sktimeLSTMFCN": True,
    "sktimeInceptionTime": True,
    "sktimeCNN": True,
    "sktimeResNet": True,
    "sktimeFCN": True,
    }


def experiment_HNEI_censoring(method_name,dataset_name="CENS_H_",censored_sources=2,seed=1,mlflow_port=5010):
    dataset_name=dataset_name+str(censored_sources)+"_seed"+str(seed)

    dataset, test_dataset = load_HNEI_censored(keep_identifiers=keep_identifiers_dict[method_name],censore_sources=censored_sources,seed=seed,rul_SA="rul")
    param_space_dict_per_method = param_space_configurations[method_name]
    method_class = rul_name_to_class[method_name]

    pre_run = HNEI_rul_best_configurations[method_name]

    run_rul_train_val_test(dataset, test_dataset, method_class, param_space_dict_per_method, method_name,
                           pre_run=pre_run, dataset_name=dataset_name, mlflow_port=mlflow_port,debug_test=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run censored RUL experiment"
    )

    parser.add_argument(
        "--level",
        type=str,
        choices=["1", "2", "3"],
        required=True,
        help="1 for 25% censoring, 2 for 50% censoring, 3 for 75% censoring"
    )

    parser.add_argument(
        "--dataset_version",
        type=str,
        choices=["1", "2", "3", "4", "5"],
        required=True,
        help="for the experiment"
    )

    parser.add_argument(
        "--method_name",
        type=str,
        choices=list(rul_name_to_class.keys()),
        required=True,
        help="Method name (must be a key of name_to_class)"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    method_name = args.method_name
    version=args.dataset_version
    level=int(args.level)*2
    seeds=[0,2,3,4,6]
    seed=seeds[int(version)-1]
    experiment_HNEI_censoring(method_name=method_name, dataset_name="CENS_H_", censored_sources=level, seed=seed, mlflow_port=None)



