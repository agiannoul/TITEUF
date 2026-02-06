import argparse

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



to_keep_identifiers={
    "DeepHit": False,
    "CoxPH": False,
    "RDSM": True,
    "RSF": False,
    "GradientBoosting": False,
    # "CNNDeepHIT": True,
}


def run_train_val_test(dataset,test_dataset,method_class,param_space_dict_per_method,method_name,
                           preprocessor=None,pre_run=None,thresholder=None,
                           additional_params={},debug=False,datasetname="",debug_TEST=True):
    experiments = [Supervised_SA_PdMExperiment]
    experiment_names = [f'SA {datasetname} Train-Val']


    methods = [method_class]

    method_names = [method_name]


    from OnlineADEngine.preprocessing.record_level.default import DefaultPreProcessor

    if preprocessor is None:
        preprocessor = DefaultPreProcessor

    if thresholder is None:
        from OnlineADEngine.thresholding.SurvSuperVisedTH import SurvToRUL
        thresholder = SurvToRUL

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
        additional_params = {
            "thresholder_threshold_value": [None]
        }
        params=run_experiment(dataset, methods, param_space_dict_per_method, method_names,
                                            experiments, experiment_names,preprocessor=preprocessor,mlflow_port=None,
                                            MAX_RUNS=8, MAX_JOBS=1, INITIAL_RANDOM=1,optimization_param="IBS",
                          debug=debug,maximize=False,thresholder=thresholder,additional_parameters=additional_params)
    best_parames= params[0]
    print(f"Best parameters: {best_parames['best_params']}")
    experiment_names = [f'SA {datasetname}']

    test_params = {}
    for key in param_space_dict_per_method[0]:
        test_params[key] = [best_parames['best_params'][f'method_{key}']]
    additional_params['thresholder_threshold_value'] = [best_parames['th_to_rul']]
    test_params["save_model"] = [True]
    method_names = [f"{method_name}"]
    print(f"test params {test_params}")

    run_experiment(test_dataset, methods, [test_params], method_names,
                   experiments, experiment_names, preprocessor=preprocessor, mlflow_port=None,
                   thresholder=thresholder, additional_parameters=additional_params,
                   MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1, optimization_param="IBS", debug=debug_TEST, maximize=False)

def experiment_HNEI_censoring(method_name,dataset_name="CENS_H_",censored_sources=2,seed=1,mlflow_port=5010):
    dataset_name=dataset_name+str(censored_sources)+"_seed"+str(seed)

    dataset, test_dataset = load_HNEI_censored(keep_identifiers=to_keep_identifiers[method_name],censore_sources=censored_sources,seed=seed,rul_SA="sa")
    param_space_dict_per_method = HNEI_SA_best_configurations[method_name]
    method_class = SA_name_to_class[method_name]


    run_train_val_test(dataset, test_dataset, method_class, param_space_dict_per_method, method_name,
                       preprocessor=None, pre_run=None, thresholder=None, datasetname=dataset_name,debug_TEST=False)



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
        choices=list(SA_name_to_class.keys()),
        required=True,
        help="Method name (must be a key of name_to_class)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    method_name = args.method_name
    version = args.dataset_version
    level = int(args.level) * 2
    seeds = [0, 2, 3, 4, 6]
    seed = seeds[int(version) - 1]
    experiment_HNEI_censoring(method_name=method_name, dataset_name="CENS_H_", censored_sources=level, seed=seed,
                              mlflow_port=None)

    # for method in SA_name_to_class.keys():
    #
    #     # 25%
    #     # experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=2, seed=0, mlflow_port=5011)
    #     # experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=2, seed=2, mlflow_port=5011)
    #     # experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=2, seed=4, mlflow_port=5011)
    #     experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=2, seed=6, mlflow_port=5011)
    #     experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=2, seed=3, mlflow_port=5011)
    #
    #     # 50%
    #     # experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=4, seed=0, mlflow_port=5011)
    #     # experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=4, seed=2, mlflow_port=5011)
    #     # experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=4, seed=4, mlflow_port=5011)
    #     experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=4, seed=6, mlflow_port=5011)
    #     experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=4, seed=3, mlflow_port=5011)
    #
    #     # 75%
    #     # experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=6, seed=0, mlflow_port=5011)
    #     # experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=6, seed=2, mlflow_port=5011)
    #     # experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=6, seed=4, mlflow_port=5011)
    #     experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=6, seed=6, mlflow_port=5011)
    #     experiment_HNEI_censoring(method_name=method, dataset_name="CENS_H_", censored_sources=6, seed=3, mlflow_port=5011)


