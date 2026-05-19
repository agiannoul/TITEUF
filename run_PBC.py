from OnlineADEngine.thresholding.constant import ConstantThresholder

from OnlineADEngine.experiment.batch.SA_experiment import Supervised_SA_PdMExperiment
from OnlineADEngine.RunExperiment import run_experiment
from OnlineADEngine.experiment.batch.RUL_experiment import SupervisedRULPdMExperiment

from utils import load_PBC

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
            "seq_len":[3,5],
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
          "n_features_per_kernel": [4], "seq_length": [1, 3, 5],"n_jobs":[8],"normalize": [False, True],}],
    "sktimeLSTMFCN": [{"n_epochs": [50, 100], "batch_size": [64, 128],
          "random_state": [42], "dropout": [0.8], "kernel_sizes": [(8, 5, 3)],
          "filter_sizes": [(128, 256, 128)], "lstm_size": [8], "seq_length": [3, 5],
          "optimizer": ["adam"], "learning_rate": [1e-2,1e-1],
          "verbose": [True],"normalize": [False,True], }],
# "sktimeLSTMFCN": [{"n_epochs": [50, 100], "batch_size": [64, 128],
#           "random_state": [42], "dropout": [0.8], "kernel_sizes": [(8, 5, 3)],
#           "filter_sizes": [(128, 256, 128)], "lstm_size": [8], "seq_length": [2, 3, 5],
#           "optimizer": ["adam"], "learning_rate": [0.001,0.01],
#           "verbose": [True],"normalize": [False], }],
    "sktimeInceptionTime": [{"n_epochs": [50, 100], "batch_size": [64],
          "random_state": [42], "kernel_size": [40],
          "n_filters": [32], "use_residual": [True],
          "optimizer": ["adam"], "learning_rate": [1e-5, 5e-5, 1e-4,1e-3],
          "use_bottleneck": [True], "bottleneck_size": [32], "depth": [6], "seq_length": [2, 3, 5],
          "verbose": [True],"normalize": [False, True], }],
    "sktimeCNN":[{"kernel_size": [3,5, 7],
                  "n_epochs": [20, 50,80,100],
                  "n_conv_layers": [2],
          "batch_size": [64, 128], "avg_pool_size": [2,3],
                  "random_state": [42]
          ,"optimizer": ["adam"],
          "learning_rate": [1e-2,0.1],
          # "learning_rate": [1e-2],
          "loss": ["mean_absolute_error"],
          "seq_length": [2, 3, 5],
          "verbose": [True],
        "normalize": [False, True],
          }],
    "sktimeResNet":[{"n_epochs": [20, 50,100], "batch_size": [64, 128],
          "random_state": [42],
          "verbose": [True], "optimizer": ["adam"],
          "learning_rate": [1e-5, 5e-5, 1e-4,1e-3], "seq_length": [2, 3, 5],
                     "normalize": [False, True],}],
    "sktimeFCN":[{"n_epochs": [30, 50,100],
          # [{"n_epochs": [5],
          "batch_size": [64],
          "random_state": [42], "seq_length": [3, 5],
          "verbose": [True], "optimizer": ["adam"],
          "learning_rate": [1e-3,1e-2,0.1],
          "activation": ["relu"],
          "normalize": [False, True],
          }],
    "GradientBoosting":[{
        "loss":['coxph'],
        "learning_rate":[0.1,0.25,0.5,0.65,0.8],
        "n_estimators":[20,30,40],
        "min_samples_split":[10,20,30],
        "min_samples_leaf":[10,20,30],
        "subsample":[0.5],
        "random_state":[42],
        "max_features":['sqrt'],
        "u_sample_rate":[0.1]
    }],
    "RSF":[{
        'n_estimators': [10,20,30,40,50],
        'min_samples_split': [5,10,15],
        'min_samples_leaf': [5,10,15],
        'max_features': ['sqrt'],
        'n_jobs': [4],
        'random_state': [42],
        'verbose': [1]
    }],
    "CoxPH":[{
        'alpha': [0.1],
        'ties': ['breslow', 'efron'],
        'n_iter': [100,150],
        'tol': [1e-8],
        'verbose': [1]
    }],
    "DeepHit":[{
        "num_nodes":[[128],[64],[32],[32, 32],[32,16,32],[64,32]],
        "batch_norm":[True,False],
        "dropout":[0,0.1],
        "batch_size":[256],
        "learning_rate":[0.0001,0.0005,0.001,0.01],
        "epochs":[100,200,300]
    }],
    "RDSM":[{'learning_rate': [0.1,1e-2, 5e-3, 1e-3, 1e-4],
            'k': [1,3,5],
            'layers': [1,2],
            'hidden': [30,40,50],
            'batch_size': [256],
            'typ': ["GRU"],
            'to_scale': [False],
            'iters': [30, 50, 100]
            # 'iters': [5]
             }],
}

def run_train_val_test(dataset,test_dataset,method_class,param_space_dict_per_method,method_name,
                           preprocessor=None,pre_run=None,thresholder=None,
                           additional_params={},debug=False,datasetname="",debug_TEST=False,optimization_param="IBS"):
    if optimization_param=="IBS":
        experiments = [Supervised_SA_PdMExperiment]
        experiment_names = [f'SA {datasetname} Train-Val']
    else:
        experiments = [SupervisedRULPdMExperiment]
        experiment_names = [f'RUL {datasetname} Train-Val']


    methods = [method_class]

    method_names = [method_name]


    from OnlineADEngine.preprocessing.record_level.default import DefaultPreProcessor

    if preprocessor is None:
        preprocessor = DefaultPreProcessor

    if thresholder is None:
        if optimization_param=="IBS":
            from OnlineADEngine.thresholding.SurvSuperVisedTH import SurvToRUL
            thresholder = SurvToRUL
        else:
            thresholder=ConstantThresholder
    if pre_run is not None:
        correct_pre_run = {}
        for key in pre_run:
            if key.startswith("method_"):
                correct_pre_run[key]=pre_run[key]
            else:
                correct_pre_run[f"method_{key}"]=pre_run[key]
        if "thresholder_threshold_value"  in additional_params.keys():
             params=[{'best_params': correct_pre_run, 'best_objective': None, 'th_to_rul':additional_params["thresholder_threshold_value"][0]}]
        else:
            params=[{'best_params': correct_pre_run, 'best_objective': None}]
        print(f"PRE RUN MODE: {pre_run}")
    else:
        if optimization_param=="IBS":
            additional_params = {
                "thresholder_threshold_value": [None]
            }
        params=run_experiment(dataset, methods, param_space_dict_per_method, method_names,
                                            experiments, experiment_names,preprocessor=preprocessor,mlflow_port=None,
                                            MAX_RUNS=20, MAX_JOBS=1, INITIAL_RANDOM=1,optimization_param=optimization_param,
                          debug=debug,maximize=False,thresholder=thresholder,additional_parameters=additional_params)
    best_parames= params[0]
    print(f"Best parameters: {best_parames['best_params']}")

    if optimization_param=="IBS":
        experiment_names = [f'SA {datasetname}']
    else:
        experiment_names = [f'RUL {datasetname}']
    test_params = {}
    for key in param_space_dict_per_method[0]:
        test_params[key] = [best_parames['best_params'][f'method_{key}']]
    if optimization_param=="IBS":
        additional_params['thresholder_threshold_value'] = [best_parames['th_to_rul']]
    test_params["save_model"] = [False]
    method_names = [f"{method_name}"]
    print(f"test params {test_params}")

    run_experiment(test_dataset, methods, [test_params], method_names,
                   experiments, experiment_names, preprocessor=preprocessor, mlflow_port=None,
                   thresholder=thresholder, additional_parameters=additional_params,
                   MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1, optimization_param=optimization_param, debug=debug_TEST, maximize=False)
keep_identifiers_dict={
    "TABPFNRv2": False,
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
    "RDSM": True,
    }

def experiment_PBC_rul(method_name,dataset_name="PBC",mlflow_port=None,path="Data/"):
    from run_rul_Best import rul_name_to_class
    keep_identifiers = keep_identifiers_dict[method_name]

    dataset, test_dataset = load_PBC(keep_identifiers=keep_identifiers,rul_sa="rul",path=path)
    param_space_dict_per_method = param_space_configurations[method_name]
    method_class = rul_name_to_class[method_name]

    pre_run = None

    run_train_val_test(dataset, test_dataset, method_class, param_space_dict_per_method, method_name,
                           pre_run=pre_run, datasetname=dataset_name,optimization_param="mape")


def experiment_PBC_sa(method_name,dataset_name="PBC",mlflow_port=None,path="Data/"):
    from runt_best_SA import SA_name_to_class
    keep_identifiers = keep_identifiers_dict.get(method_name, False)
    dataset, test_dataset = load_PBC(keep_identifiers=keep_identifiers,rul_sa="sa",path=path)
    param_space_dict_per_method = param_space_configurations[method_name]
    method_class = SA_name_to_class[method_name]

    pre_run = None

    run_train_val_test(dataset, test_dataset, method_class, param_space_dict_per_method, method_name,
                           pre_run=pre_run, datasetname=dataset_name,optimization_param="IBS")


if __name__ == "__main__":
    method_name = "sktimeLSTMFCN"
    experiment_PBC_rul(method_name=method_name,dataset_name="PBC",path="Data/")
    # method_name = "GradientBoosting"
    # experiment_PBC_sa(method_name=method_name,dataset_name="PBC",path="Data/")