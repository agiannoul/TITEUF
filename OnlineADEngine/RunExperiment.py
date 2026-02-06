from OnlineADEngine.pipeline.pipeline import PdMPipeline
from OnlineADEngine.preprocessing.record_level.default import DefaultPreProcessor
from OnlineADEngine.postprocessing.default import DefaultPostProcessor
from OnlineADEngine.thresholding.constant import ConstantThresholder
from OnlineADEngine.utils.utils import calculate_mango_parameters

from OnlineADEngine.experiment.batch.RUL_experiment import SupervisedRULPdMExperiment
from OnlineADEngine.experiment.batch.SA_experiment import Supervised_SA_PdMExperiment

from OnlineADEngine.method.supervised_method import SupervisedMethodInterface

import socket
import subprocess


def get_method_type(experiment):
    if experiment == SupervisedRULPdMExperiment:
        return SupervisedMethodInterface
    elif experiment == Supervised_SA_PdMExperiment:
        return SupervisedMethodInterface
    raise ValueError(f"Unknown experiment type: {experiment}.")
def is_port_in_use(host, port):
    """Check if a given port is in use on the specified host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def run_mlflow_server(mlflow_port):
    """Function to run MLflow server if it's not already running."""
    host = "127.0.0.1"
    port = mlflow_port

    if is_port_in_use(host, port):
        print(f"MLflow server is already running at http://{host}:{port}.")
    else:
        print("Starting MLflow server...")
        # subprocess.Popen(["export","MLFLOW_TRACKING_URI=sqlite:///mlruns.db"])
        subprocess.Popen(["mlflow", "ui", "--host", host, "--port", str(port)])
        print(f"MLflow server started at http://{host}:{port}.")


def run_experiment(dataset,methods, param_space_dict_per_method,method_names,experiments,
                   experiment_names,additional_parameters={},MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1,profile_size=2,
                   fit_size=None,postprocessor=DefaultPostProcessor,preprocessor = DefaultPreProcessor,
                   thresholder=ConstantThresholder,mlflow_port=None,debug=True,optimization_param="AD1_AUC",maximize=True):
    """

    :param dataset: A dictionary with dataset properties (look ad loadDataset.py)
    :param methods: List of methods to run
    :param param_space_dict_per_method: List with dictionaries, each dictionary has the parameter space to be tested for each method
    :param method_names: List with the names of methods to run
    :param experiments: List with Experiment objects (e.g. AutoProfileSemiSupervisedPdMExperiment)
    :param experiment_names: Name of each experiment. This is combined with method names and used for logging purposes
    :param additional_parameters: dictionary containing paremeters for postprocessor,preprocessor,thresholder. Keys should start from 'postprocessor_', 'preprocessor_', 'thresholder_'.
    :param MAX_RUNS: Maximum number of runs to run
    :param MAX_JOBS: Cores to use for parameter searching
    :param INITIAL_RANDOM: initial random runs
    :param profile_size: Profile size is used in AutoProfileSemiSupervisedPdMExperiment
    :param fit_size: This indicates the Initial Profile size is used in AutoProfileSemiSupervisedPdMExperiment
    :param postprocessor: Postprocessing to apply
    :param preprocessor: Preprocessing to apply
    :param thresholder: The thresholding technique to use (default constant)
    :param mlflow_port: The port to run mlflow (in case of None it does not run)
    :return: best parameterization
    """

    all_experiments_best_parameters = []
    for current_method, current_method_param_space, current_method_name in zip(methods, param_space_dict_per_method,
                                                                               method_names):


        if mlflow_port is not None:
            run_mlflow_server(mlflow_port)
        for experiment, experiment_name in zip(experiments, experiment_names):
            current_param_space_dict = {

            }


            my_pipeline = PdMPipeline(
                steps={
                    'preprocessor': preprocessor,
                    'method': current_method,
                    'postprocessor': postprocessor,
                    'thresholder': thresholder,
                },
                dataset=dataset,
                auc_resolution=30,
                experiment_type=get_method_type(experiment)
            )


            for key, value in current_method_param_space.items():
                current_param_space_dict[f'method_{key}'] = value
            for key, value in additional_parameters.items():
                current_param_space_dict[key] = value

            num, jobs, initial_random = calculate_mango_parameters(current_param_space_dict, MAX_JOBS, INITIAL_RANDOM,
                                                                   MAX_RUNS)
            constraint_function=None
            my_experiment = experiment(
                experiment_name=experiment_name + ' ' + current_method_name,
                target_data=dataset['target_data'],
                target_sources=dataset['target_sources'],
                pipeline=my_pipeline,
                param_space=current_param_space_dict,
                num_iteration=num,
                n_jobs=jobs,
                initial_random=initial_random,
                artifacts='./artifacts/' + experiment_name + ' artifacts',
                constraint_function=constraint_function,
                debug=debug,
                optimization_param=optimization_param,
                maximize=maximize
            )


            best_params = my_experiment.execute()
            print(experiment_name)
            print(best_params)
            all_experiments_best_parameters.append(best_params)
    return all_experiments_best_parameters