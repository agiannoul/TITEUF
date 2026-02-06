from models.xgboostRUL import XGBoostRUL
from utils import load_test, run_rul_train_val_test, load_FEMTO_rul, load_HNEI_rul, load_XJTU
from models.sktime_wrapper import FCNRegressorRUL
from models.sktime_wrapper import ResNetRegressorRUL
from models.sktime_wrapper import LSTMFCNRegressorRUL
from models.sktime_wrapper import CNNTRegressorRUL
from models.sktime_wrapper import InceptionTimeRegressorRUL
from models.sklearn_wraper import ElasticNetRUL
from models.sklearn_wraper import RandomForestRUL
from utils import read_azure_rul


def experiment_scania(method_name,load_data_method=load_test,dataset_name="SCANIA",mlflow_port=5011):
    dataset, test_dataset =  load_data_method(keep_identifiers=keep_identifiers_dict[method_name])
    param_space_dict_per_method = param_space_configurations[method_name]
    method_class=name_to_class[method_name]

    pre_run = None

    run_rul_train_val_test(dataset,test_dataset,method_class,param_space_dict_per_method,method_name,
                           pre_run=pre_run, dataset_name=dataset_name,mlflow_port=mlflow_port)

def experiment_azure(method_name,load_data_method=read_azure_rul,dataset_name="Azure",mlflow_port=5011):
    dataset, test_dataset =  load_data_method(keep_identifiers=keep_identifiers_dict[method_name])
    param_space_dict_per_method = param_space_configurations[method_name]
    method_class=name_to_class[method_name]

    pre_run = None

    run_rul_train_val_test(dataset,test_dataset,method_class,param_space_dict_per_method,method_name,
                           pre_run=pre_run,dataset_name=dataset_name,mlflow_port=mlflow_port)

# def experiment_femto(method_name,load_data_method=load_FEMTO_rul,dataset_name="FEMTO",mlflow_port=5011):
#     dataset, test_dataset = load_data_method(keep_identifiers=keep_identifiers_dict[method_name])
#     param_space_dict_per_method = param_space_configurations[method_name]
#     method_class = name_to_class[method_name]
#
#     pre_run = None
#
#     run_rul_train_val_test(dataset, test_dataset, method_class, param_space_dict_per_method, method_name,
#                            pre_run=pre_run, dataset_name=dataset_name, mlflow_port=mlflow_port)
#
# def experiment_XJTU(method_name,load_data_method=load_XJTU,dataset_name="XJTU",mlflow_port=5011):
#     dataset, test_dataset = load_XJTU(keep_identifiers=keep_identifiers_dict[method_name],rul_SA="rul")
#     param_space_dict_per_method = param_space_configurations[method_name]
#     method_class = name_to_class[method_name]
#
#     pre_run = None
#
#     run_rul_train_val_test(dataset, test_dataset, method_class, param_space_dict_per_method, method_name,
#                            pre_run=pre_run, dataset_name=dataset_name, mlflow_port=mlflow_port)

def experiment_HNEI(method_name,load_data_method=load_HNEI_rul,dataset_name="HNEI",mlflow_port=5011):
    dataset, test_dataset = load_data_method(keep_identifiers=keep_identifiers_dict[method_name])
    param_space_dict_per_method = param_space_configurations[method_name]
    method_class = name_to_class[method_name]

    pre_run = None

    run_rul_train_val_test(dataset, test_dataset, method_class, param_space_dict_per_method, method_name,
                           pre_run=pre_run, dataset_name=dataset_name, mlflow_port=mlflow_port)


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

param_space_configurations={
    "TABPFNv2":[{"default": [True],"n_samples":[800,1000]}],
    "XGBoost":[{"n_estimators":[1000,500,200,100],           # increase, but use early stopping
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
    "RandomForestRUL": [{"n_estimators": [100,200,300,500],
                                   "max_depth": [5, 10, None],
                                   "n_jobs": [8],
                                   "min_samples_leaf": [1, 5, 10],
                                   "random_state": [42],
                                   "min_samples_split": [2,5,10],
                                   "max_features": ["sqrt", "log2"],
                                   }],
    "ElasticNetRUL": [{"alpha": [0.0001,0.001, 0.01, 0.1],
                               "l1_ratio": [0.1,0.2, 0.5,0.6, 0.8],
                               "max_iter": [500, 1000, 2000,3000],
                               "random_state": [42],
                               "method_selection": ["cyclic", "random"]
                               }],
    "sktimeLSTMFCN": [{"n_epochs": [50,80,100,150], "batch_size": [64, 128],
          "random_state": [42], "dropout": [0.8], "kernel_sizes": [(8, 5, 3)],
          "filter_sizes": [(128, 256, 128)], "lstm_size": [8], "seq_length": [5, 10, 20],
          "optimizer": ["adam"],
                       "learning_rate": [1e-5, 1e-4,1e-3,1e-2],
                  "verbose": [True],
                "normalize": [False,True],

                }],
    "sktimeInceptionTime": [{"n_epochs": [50,100,150], "batch_size": [64],
          "random_state": [42], "kernel_size": [40],
          "n_filters": [32], "use_residual": [False,True],
          "optimizer": ["adam"], "learning_rate": [1e-5, 5e-5, 1e-4,1e-3],
          "use_bottleneck": [True], "bottleneck_size": [32], "depth": [6], "seq_length": [5, 10, 20],
          "verbose": [True],
            "normalize": [False,True],
                             }],
    "sktimeCNN":[{"kernel_size": [5, 7],
                  "n_epochs": [20, 50,80,100],
                  "n_conv_layers": [2],
          "batch_size": [64, 128], "avg_pool_size": [2],#[3],
                  "random_state": [42]
          ,"optimizer": ["adam"],
          "learning_rate": [1e-5, 5e-5, 1e-4,1e-3,1e-2],
          "loss": ["mean_absolute_error"],
          "seq_length": [5, 10, 20],
          "verbose": [True],
                  "normalize": [False,True],
          }],
    "sktimeResNet":[{"n_epochs": [20,50,100,150], "batch_size": [64, 128],
          "random_state": [42],
          "verbose": [True], "optimizer": ["adam"],
          "learning_rate": [1e-5, 5e-5, 1e-4,1e-3], "seq_length": [5, 10, 20],
                     "normalize": [False,True],
                     }],

    "sktimeFCN":[{"n_epochs": [30, 50,100,150],
          # [{"n_epochs": [5],
          "batch_size": [64],
          "random_state": [42], "seq_length": [5, 10,20],
          "verbose": [True], "optimizer": ["adam"],
          "learning_rate": [1e-5, 5e-5, 1e-4,1e-3],
          "activation": ["relu"],
          "normalize": [False,True],
          }],
    }

from models.tabpfnreg import TABPFNRUL
# from models.CatBoost_W import CatBoostWRUL

name_to_class={
    "TABPFNv2":TABPFNRUL,
    "XGBoost":XGBoostRUL,
    # "CatBoost_W_RUL":CatBoostWRUL,
    "RandomForestRUL":RandomForestRUL,
    "ElasticNetRUL":ElasticNetRUL,
    "sktimeLSTMFCN":LSTMFCNRegressorRUL,
    "sktimeInceptionTime":InceptionTimeRegressorRUL,
    "sktimeCNN":CNNTRegressorRUL,
    "sktimeResNet":ResNetRegressorRUL,
    "sktimeFCN":FCNRegressorRUL,
    }

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PdM experiments"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["Azure", "SCANIA","XJTU","HNEI"],
        required=True,
        help="Dataset to run the experiment on"
    )

    parser.add_argument(
        "--method_name",
        type=str,
        choices=list(name_to_class.keys()),
        required=True,
        help="Method name (must be a key of name_to_class)"
    )

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "Azure":
        experiment_azure(method_name=args.method_name,mlflow_port=None)
    elif args.dataset == "SCANIA":
        experiment_scania(method_name=args.method_name,mlflow_port=None)
    elif args.dataset == "HNEI":
        experiment_HNEI(method_name=args.method_name,mlflow_port=None)