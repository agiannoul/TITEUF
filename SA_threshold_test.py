import numpy as np
import pandas as pd
from OnlineADEngine.RunExperiment import run_experiment
from OnlineADEngine.experiment.batch.SA_experiment import Supervised_SA_PdMExperiment
from OnlineADEngine.method.supervised_method import SupervisedMethodInterface
from OnlineADEngine.pdm_evaluation_types.types import EventPreferences
from sksurv.ensemble import RandomSurvivalForest

from models.CoxModel import CoxPH
from models.DeepHit import DeepHIT
from models.GradientBoosting import GradientBoostingSurvival
from models.RDSMmodel import RDSM
from utils import load_HNEI_censored


class RSF(SupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences, save_model=False, *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.model_per_source = {}
        self.avail_times_per_source = {}
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.save_model = save_model

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
            ydf=pd.DataFrame({'event': [lb[1] for lb in labels],'RUL': [lb[0] for lb in labels]})
            y = Surv.from_dataframe("event", "RUL", ydf)

            # RandomSurvivalForest(n_estimators=100, min_samples_split=6, min_samples_leaf=5, verbose=1, n_jobs=4)
            self.model_per_source[current_historic_source] = RandomSurvivalForest(*self.initial_args,**self.initial_kwargs)
            self.model_per_source[current_historic_source].fit(current_historic_data, y)
            self.avail_times_per_source[current_historic_source]=np.unique([ty for ty in y['RUL']])

            if self.save_model:
                import pickle
                with open(f"xgboost_model_{current_historic_source}.pkl", "wb") as f:
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
        return "RSF"

    def get_all_models(self):
        pass


SA_name_to_class={
    "CoxPH": CoxPH,
    "DeepHit": DeepHIT,
    "RDSM": RDSM,
    "RSF": RSF,
    "GradientBoosting": GradientBoostingSurvival,
}


HNEI_SA_best_configurations={
    "DeepHit": [{'batch_norm': [False], 'batch_size': [256], 'dropout': [0.1],
            'epochs': [300], 'learning_rate': [0.01], 'num_nodes': [[64]]}],
"GradientBoosting": [{"loss": ['coxph'],
    "learning_rate": [0.5],
    "n_estimators": [50],
    "min_samples_split": [10],
    "min_samples_leaf": [1],
    "random_state": [42]}],
"CoxPH":[ {
    'alpha': [0.1],
    'ties': ['efron'],
    'n_iter': [100],
    'tol': [1e-8],
    'verbose': [1]
}],
"RDSM":[ {
    'batch_size': [256], 'hidden': [40], 'iters': [100], 'k': [1], 'layers': [2],
    'learning_rate': [0.01], 'typ': ['GRU'],
}],
"RSF": [{
    'n_estimators': [30],
    'min_samples_split': [10],
    'min_samples_leaf': [10],
    'max_features': ['sqrt'],
    'n_jobs': [4],
    'random_state': [42],
    'verbose': [1]
}],
}

scania_best_configurations={
    "DeepHit":{'batch_norm': True, 'batch_size': 512, 'dropout': 0.1,
             'epochs': 200, 'learning_rate': 0.001, 'num_nodes': [32],},
    "GradientBoosting":{"loss":'coxph',
        "learning_rate":0.5,
        "n_estimators":30,
        "min_samples_split":10,
        "min_samples_leaf":1,
        "random_state":42,
        "u_sample_rate":0.3},
    "CoxPH":{
            'alpha': 0.1,
            'ties': 'breslow',
            'n_iter': 100,
            'tol': 1e-8,
            'verbose': 1
        },
    "RDSM":{
        'batch_size': 128, 'hidden': 100, 'iters': 50, 'k': 1, 'layers': 5,
        'learning_rate': 0.01, 'typ': 'GRU',
    },
    "RSF": {
        'n_estimators': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'n_jobs': 4,
        'random_state': 42,
        'verbose': 1
    },
}

azure_best_configurations={
    "DeepHit":{'batch_norm': True, 'batch_size': 256, 'dropout': 0.1,
             'epochs': 200, 'learning_rate': 0.001, 'num_nodes': [32],},
    "CoxPH":{
            'alpha': 0.1,
            'ties': 'efron',
            'n_iter': 150,
            'tol': 1e-9,
            'verbose': 1
        },
    "RDSM":{
        'batch_size': 256, 'hidden': 30, 'iters': 50, 'k': 1, 'layers': 2,
        'learning_rate': 0.01, 'typ': 'GRU',
    },
    "RSF": {
        'n_estimators': 20,
        'min_samples_split': 10,
        'min_samples_leaf': 15,
        'max_features': 'sqrt',
        'n_jobs': 4,
        'random_state': 42,
        'verbose': 1
    },
"GradientBoosting":{"loss":'coxph',
        "learning_rate":0.5,
        "n_estimators":30,
        "min_samples_split":30,
        "min_samples_leaf":1,
        "random_state":42,
        "u_sample_rate":0.1},
}



to_keep_identifiers={
    "DeepHit": False,
    "CoxPH": False,
    "RDSM": True,
    "RSF": False,
    "GradientBoosting": False,
    # "CNNDeepHIT": True,
}


def run_train_val_test05(dataset,test_dataset,method_class,param_space_dict_per_method,method_name,
                           preprocessor=None,pre_run=None,thresholder=None,
                           additional_params={},debug=False,datasetname="",debug_TEST=True):
    experiments = [Supervised_SA_PdMExperiment]
    experiment_names = [f'SA {datasetname}']


    methods = [method_class]

    method_names = [method_name]


    from OnlineADEngine.preprocessing.record_level.default import DefaultPreProcessor

    if preprocessor is None:
        preprocessor = DefaultPreProcessor

    if thresholder is None:
        from OnlineADEngine.thresholding.SurvSuperVisedTH import SurvToRUL
        thresholder = SurvToRUL


    additional_params = {
        "thresholder_threshold_value": [0.5]
    }
    params=run_experiment(test_dataset, methods, param_space_dict_per_method, method_names,
                                        experiments, experiment_names,preprocessor=preprocessor,mlflow_port=None,
                                        MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1,optimization_param="IBS",
                      debug=debug_TEST,maximize=False,thresholder=thresholder,additional_parameters=additional_params)




def experiment_HNEI(method_name,dataset_name="HNEI",censored_sources=0,seed=0):
    dataset, test_dataset = load_HNEI_censored(keep_identifiers=to_keep_identifiers[method_name],censore_sources=censored_sources,seed=seed,rul_SA="sa")
    param_space_dict_per_method = HNEI_SA_best_configurations[method_name]
    method_class = SA_name_to_class[method_name]
    print(param_space_dict_per_method)
    run_train_val_test05(dataset, test_dataset, method_class, param_space_dict_per_method, method_name,
                       preprocessor=None, pre_run=None, thresholder=None, datasetname=dataset_name)

from utils import load_SACNIA_surv
from utils import read_azure

def experiment_SCANIA(method_name,dataset_name="SCANIA05"):
    dataset, test_dataset = load_SACNIA_surv(keep_identifiers=to_keep_identifiers[method_name])
    temp_dict=scania_best_configurations[method_name]
    for key in temp_dict.keys():
        temp_dict[key]=[temp_dict[key]]
    param_space_dict_per_method = [temp_dict]
    method_class = SA_name_to_class[method_name]
    print(param_space_dict_per_method)
    run_train_val_test05(dataset, test_dataset, method_class, param_space_dict_per_method, method_name,
                       preprocessor=None, pre_run=None, thresholder=None, datasetname=dataset_name)

def experiment_Azure(method_name,dataset_name="AZURE05"):
    dataset, test_dataset = read_azure(keep_identifiers=to_keep_identifiers[method_name])
    temp_dict=azure_best_configurations[method_name]
    for key in temp_dict.keys():
        temp_dict[key]=[temp_dict[key]]
    param_space_dict_per_method = [temp_dict]
    method_class = SA_name_to_class[method_name]
    print(param_space_dict_per_method)
    run_train_val_test05(dataset, test_dataset, method_class, param_space_dict_per_method, method_name,
                       preprocessor=None, pre_run=None, thresholder=None, datasetname=dataset_name)


import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run using threshold equal to 0.5 on SA methods"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["Azure", "SCANIA","HNEI"],
        required=True,
        help="Dataset to run the experiment on"
    )

    parser.add_argument(
        "--method_name",
        type=str,
        choices=list(SA_name_to_class.keys()),
        required=True,
        help="Method name (must be a key of name_to_class)"
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    if args.dataset == "Azure":
        experiment_Azure(args.method_name, dataset_name="AZURE05")
    elif args.dataset == "SCANIA":
        experiment_SCANIA(args.method_name, dataset_name="SCANIA05")
    elif args.dataset == "HNEI":
        experiment_HNEI(args.method_name, dataset_name="HNEI05")
    else:
        print("Invalid dataset name. Please choose from 'Azure', 'SCANIA', or 'HNEI'.")

    # experiment_Azure("GradientBoosting", dataset_name="AZURE05")
    # experiment_SCANIA("GradientBoosting", dataset_name="SCANIA05")
    # experiment_HNEI("GradientBoosting", dataset_name="HNEI05")

