import logging
import os
import math
import abc

import random
import re
from typing import Callable
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import uuid

from matplotlib import cm

from OnlineADEngine.pipeline.pipeline import PdMPipeline
from OnlineADEngine.utils.rul_transformations import hard_transform_survival, sigmoid_survival_batch

logging.basicConfig(level=logging.INFO)


def process_data(current_data, header, data_type) -> list[pd.DataFrame]:
    if len(current_data) == 0:
        return current_data
    if isinstance(current_data, pd.DataFrame):
        result = [current_data]
    elif isinstance(current_data, str):
        # if it is a string check if it is a csv file or directory containing csv files
        if current_data.endswith('.csv'):
            result = [pd.read_csv(current_data, header=header)]
        elif Path(current_data).is_dir():
            result = []

            current_directory_files = os.listdir(current_data)
            current_csv_files = [file for file in current_directory_files if file.endswith('.csv')]
            for csv_file in current_csv_files:
                current_csv_file_path = os.path.join(current_data, csv_file)
                result.append(pd.read_csv(current_csv_file_path, header=header))
    elif isinstance(current_data, list):
        result = current_data
        # necessarily nested in order to avoid exception when looping on a variable that is not a list because python does not support short-circuit evaluation
        if not all(isinstance(item, pd.DataFrame) for item in current_data):
            raise Exception(f'Some element of the list parameter \'{data_type}\' has unsupported type')
    else:
        raise Exception(f'Not supported type {type(current_data)} for parameter \'{data_type}\'')

    return result


class PdMExperiment(abc.ABC):
    def __init__(self,
                 experiment_name: str,
                 pipeline: PdMPipeline,
                 param_space: dict,
                 constraint_function: Callable = None,
                 target_data: list[pd.DataFrame] = None,
                 # TODO str for directory with csv files for each scenario or single csv file of one scenario
                 target_sources: list[str] = None,
                 historic_data: list[pd.DataFrame] = [],
                 # TODO str for directory with csv files for each scenario or single csv file of one scenario
                 historic_sources: list[str] = [],
                 optimization_param: str = 'AD1_AUC',
                 initial_random: int = 2,
                 num_iteration: int = 20,
                 batch_size: int = 1,
                 n_jobs: int = 1,
                 random_state: int = 42,
                 random_n_tries: int = 3,
                 constraint_max_retries: int = 10,
                 historic_data_header: str = 'infer',
                 target_data_header: str = 'infer',
                 artifacts: str = 'artifacts',
                 debug: bool = False,
                 delay: float = None,  # in milliseconds
                 log_best_scores: bool = False,
                 maximize: bool = True
                 ):
        self.experiment_name = experiment_name
        # TODO target and historic data and sources parameter should be removed, became default parameters for backwards compatibility
        self.historic_data = pipeline.dataset['historic_data']
        self.historic_sources = pipeline.dataset['historic_sources']
        self.target_data = pipeline.dataset['target_data']
        self.target_sources = pipeline.dataset['target_sources']
        self.pipeline = pipeline
        self.param_space = param_space
        self.optimization_param = optimization_param
        self.initial_random = initial_random
        self.num_iteration = num_iteration
        self.maximize = maximize
        # self.batch_size = batch_size currently commented out because of using only scheduler.parallel, more info on issue #97 on Mango - alternatives include using only scheduler.parallel or letting the user decide depending on his hardware
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.historic_data_header = historic_data_header
        self.target_data_header = target_data_header
        self.artifacts = artifacts

        self.debug = debug
        self.delay = delay

        self.log_best_scores = log_best_scores
        current_uuid = uuid.uuid4()
        self.lock_file_path = f'pdm_evaluation_framework_lock_file_{current_uuid}.lock'
        self.best_scores_info_dict_path = f'best_scores_info_{current_uuid}.pkl'

        self.event_data = self.pipeline.event_data
        self.constraint_function = constraint_function

        # TODO the next line is probably useless
        Path(self.artifacts).mkdir(parents=True, exist_ok=True)
        self.extra_metrics = {}
        # process historic data
        self.historic_data = process_data(self.historic_data, historic_data_header, 'historic_data')

        # process target data
        self.target_data = process_data(self.target_data, target_data_header, 'target_data')

        self.experiment_id = None

        random.seed(self.random_state)

        import pkg_resources
        required = {'torch'}
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = required - installed
        if not missing:
            import torch
            torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # current_dir = os.getcwd()
        # os.chdir("./src/OnlineADEngine/evaluation/RBPR_official")

        # Run make clean

        # if os.name == 'nt':
        #     powershell_command = "(Get-Content Makefile) -replace 'rm', 'del' | Out-File -encoding ASCII Makefile"
        #     subprocess.run(["powershell", "-Command", powershell_command])
        #     subprocess.run(["powershell", "-Command", "make", "clean"])
        #     subprocess.run(["powershell", "-Command", "make"])
        # else:
        #     subprocess.call(["make","-f","MakefileL", "clean"])
        #     subprocess.call(["make","-f","MakefileL"])
        #     # Move the evaluate executable to the parent directory
        #     subprocess.call(["mv", "evaluate", ".."])
        # Run make

        # Change back to the original directory
        # os.chdir(current_dir)

    @abc.abstractmethod
    def execute(self) -> dict:
        pass

    def _register_experiment(self) -> None:
        # if self.delay is not None:
        #     print(f'Cooldown for {self.delay} milliseconds')
        #     time.sleep(self.delay / 1000)

        try:
            self.experiment_id = mlflow.create_experiment(name=self.experiment_name)
        except Exception as e:
            logging.warning(
                f'Experiment with experiment name \'{self.experiment_name}\' already exists. Be careful if you are sure about including your run in this experiment.')
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id

    def _inner_plot(self, color, rangearay, datesofscores, minvalue, maxvalue, label):
        plt.fill_between(datesofscores, minvalue, maxvalue, where=rangearay, color=color,
                         alpha=0.3, label=label)

    def _plot_SA(self, plot_dictionary) -> None:
        # plot_rul_dictionary[current_target_source]={"scores":processed_target_scores,"labels":current_labels,"thresholds":None,"index":current_dates}
        if self.debug:
            plt.figure(figsize=(20, 20))
            counter = 0
            size = 3
            globalcounter = 0
            namescount = -1

            # plot_rul_dictionary[current_target_source]={"scores":processed_target_scores,"labels":current_labels,"thresholds":None,"index":current_dates}
            for key in plot_dictionary.keys():
                counter += 1
                globalcounter += 1
                if globalcounter > 30:
                    break
                plt.subplot(size * 100 + 10 + counter)

                self._plot_SA_inner(plot_dictionary[key]["scores"], plot_dictionary[key]["labels"])
                if counter == size:
                    namescount += 1
                    mlflow.log_figure(plt.gcf(), f"scores_{namescount * 9}_{namescount * 9 + counter}.png")
                    plt.clf()
                    counter = 0
            if counter > 0:
                namescount += 1
                mlflow.log_figure(plt.gcf(), f"scores_{namescount * 9}_{namescount * 9 + counter}.png")
                plt.clf()
                counter = 0

    def _plot_SA_inner(self, scores, labels):
        pivcounter = -1
        pivot = 10
        cmap = cm.get_cmap('coolwarm')
        # Pick two colors (not the edges, e.g., 0.25 and 0.75)
        color1 = cmap(0.15)
        color2 = cmap(0.75)
        for score_length, lab in zip(scores, labels):
            pivcounter += 1
            if pivcounter % pivot != 0:
                continue
            plt.plot(score_length[1], score_length[0], color=color1)

            closest_pos = np.argmin(np.abs(np.array(score_length[1]) - lab[0]))
            color = "red" if lab[1] == 1 else "black"
            plt.scatter(score_length[1][closest_pos], score_length[0][closest_pos], color=color, zorder=3)

    def plot_SA_of_RUL(self, plot_test_preds, result_labels, is_rtf):
        if self.debug:
            plt.figure(figsize=(20, 20))
            counter = 0
            size = 3
            globalcounter = 0
            namescount = -1

            cmap = cm.get_cmap('coolwarm')
            # Pick two colors (not the edges, e.g., 0.25 and 0.75)
            color1 = cmap(0.15)
            color2 = cmap(0.75)
            # plot_rul_dictionary[current_target_source]={"scores":processed_target_scores,"labels":current_labels,"thresholds":None,"index":current_dates}
            for pred_set, lab_set, rtf in zip(plot_test_preds, result_labels, is_rtf):
                counter += 1
                globalcounter += 1
                if globalcounter > 30:
                    break
                plt.subplot(size * 100 + 10 + counter)
                self._plot_SA_inner(pred_set, [(lab, rtf) for lab in lab_set])
                if counter == size:
                    namescount += 1
                    mlflow.log_figure(plt.gcf(), f"SA_{namescount * 9}_{namescount * 9 + counter}.png")
                    plt.clf()
                    counter = 0
            if counter > 0:
                namescount += 1
                mlflow.log_figure(plt.gcf(), f"SA_{namescount * 9}_{namescount * 9 + counter}.png")
                plt.clf()
                counter = 0

    def _plot_RUL(self, plot_dictionary) -> None:
        plt.figure(figsize=(20, 20))
        if self.debug:
            counter = 0
            namescount = -1
            # plot_rul_dictionary[current_target_source]={"scores":processed_target_scores,"labels":current_labels,"thresholds":None,"index":current_dates}
            for key in plot_dictionary.keys():
                if key in ["recall", "prc", "anomaly_ranges", "lead_ranges"]:
                    continue
                else:
                    if plot_dictionary[key]["rtf"] != 1:
                        continue
                    counter += 1
                    if namescount > 40:
                        break
                    plt.subplot(910 + counter)
                    plt.plot(plot_dictionary[key]["index"], plot_dictionary[key]["scores"], ".-", color="red",
                             label="RUL predictions")
                    plt.plot(plot_dictionary[key]["index"], plot_dictionary[key]["labels"], ".-", color="black",
                             label="RUL LABELS ")
                    if counter == 9:
                        namescount += 1
                        mlflow.log_figure(plt.gcf(), f"scores_{namescount * 9}_{namescount * 9 + counter}.png")
                        plt.clf()
                        counter = 0
            if counter > 0:
                namescount += 1
                mlflow.log_figure(plt.gcf(), f"scores_{namescount * 9}_{namescount * 9 + counter}.png")
                plt.clf()
                counter = 0

    def _plot_scores(self, plot_dictionary, best_metrics_dict) -> None:
        tups = []
        for rec, prc in zip(plot_dictionary['recall'], plot_dictionary['prc']):
            tups.append((rec, prc))
        tups = sorted(tups, key=lambda x: (x[0], -x[1]))
        xaxisvalue = []
        yaxisvalue = []
        for tup in tups:
            xaxisvalue.append(tup[0])
            yaxisvalue.append(tup[1])
        plt.plot(xaxisvalue, yaxisvalue, "-o")
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), 'pr_curve.png')

        plt.clf()
        plt.figure(figsize=(20, 20))
        if self.debug:
            counter = 0
            namescount = -1
            prelimit = 0
            for key in plot_dictionary.keys():
                if key == "recall" or key == "prc" or key == "anomaly_ranges" or key == "lead_ranges":
                    continue
                counter += 1
                data_to_plot = plot_dictionary[key]
                current_range = plot_dictionary["anomaly_ranges"][prelimit:prelimit + len(data_to_plot["scores"])]
                current_range_lead = plot_dictionary["lead_ranges"][prelimit:prelimit + len(data_to_plot["scores"])]

                prelimit += len(data_to_plot["scores"])
                plt.subplot(910 + counter)
                # print()
                plt.plot(data_to_plot["index"], data_to_plot["scores"], ".-", color="black", label="anomaly score")
                plt.plot(data_to_plot["index"],
                         [best_metrics_dict["threshold_auc"] for i in range(len(data_to_plot["index"]))], ".-",
                         color="dodgerblue", label="best threshold")

                for date in data_to_plot["failures"]:
                    plt.axvline(date, color="red")

                # plot PH
                self._inner_plot("red", current_range, data_to_plot["index"], min(data_to_plot["scores"]),
                                 max(data_to_plot["scores"]), "predictive horizon")

                # plot lead
                self._inner_plot("grey", current_range_lead, data_to_plot["index"], min(data_to_plot["scores"]),
                                 max(data_to_plot["scores"]), "lead time")
                plt.legend(loc="center left")
                plt.title(f'Source label: {key}')

                if counter == 9:
                    namescount += 1
                    mlflow.log_figure(plt.gcf(), f"scores_{namescount * 9}_{namescount * 9 + counter}.png")
                    plt.clf()
                    counter = 0
            if counter > 0:
                namescount += 1
                mlflow.log_figure(plt.gcf(), f"scores_{namescount * 9}_{namescount * 9 + counter}.png")
                plt.clf()
                counter = 0

    def _finish_run(self, parent_run, current_steps) -> None:
        if 'many' in current_steps['method'].get_library():
            model_sources, models = current_steps['method'].get_all_models()
            for model_source, model in zip(model_sources, models):
                current_subpackage = getattr(mlflow, re.sub('many_', '', current_steps['method'].get_library()))
                current_submodule = current_subpackage.log_model
                # TODO do not use self.artifacts
                current_submodule(model, f'{self.artifacts}/{str(current_steps["method"])}_source_{model_source}')
        elif current_steps['method'].get_library() == 'no_save':
            pass
        else:
            # TODO we should check if there is a log_model functionality for the method we have in the current run
            # TODO do not use self.artifacts
            current_subpackage = getattr(mlflow, current_steps['method'].get_library())
            current_submodule = current_subpackage.log_model
            current_submodule(current_steps['method'], f'{self.artifacts}/{str(current_steps["method"])}')

        # log parameters for each step
        for step in self.pipeline.get_steps().keys():
            mlflow.log_params({
                f'{step}_{key}': str(value)[:499] for key, value in current_steps[step].get_params().items()
            })
            mlflow.log_param(step, str(current_steps[step]))

        if "anomaly_ranges" in self.pipeline.dataset.keys():
            mlflow.log_param('anomaly_ranges', self.pipeline.dataset['anomaly_ranges'])
            if self.pipeline.dataset["anomaly_ranges"]:
                mlflow.log_params({
                    'predictive_horizon': self.pipeline.slide,
                    'beta': self.pipeline.beta,
                    'lead': self.pipeline.slide
                })
            else:
                mlflow.log_params({
                    'predictive_horizon': self.pipeline.predictive_horizon,
                    'beta': self.pipeline.beta,
                    'lead': self.pipeline.lead
                })
        else:
            mlflow.log_params({
                'predictive_horizon': self.pipeline.predictive_horizon,
                'beta': self.pipeline.beta,
                'lead': self.pipeline.lead
            })

        for paramm in ["slide", "auc_resolution", "min_historic_scenario_len", "min_target_scenario_len",
                       "max_wait_time"]:
            if paramm in self.pipeline.dataset and paramm is not None:
                mlflow.log_param(paramm, self.pipeline.dataset[paramm])
        # mlflow.log_params({
        #     'slide': self.pipeline.dataset['slide'],
        #     'auc_resolution': self.pipeline.auc_resolution,
        #     'min_historic_scenario_len': self.pipeline.dataset['min_historic_scenario_len'],
        #     'min_target_scenario_len': self.pipeline.dataset['min_target_scenario_len'],
        #     'max_wait_time': self.pipeline.dataset['max_wait_time']
        # })

        if 'reset_after_fail' in self.pipeline.dataset:
            mlflow.log_param('reset_after_fail', self.pipeline.dataset['reset_after_fail'])

        if 'setup_1_period' in self.pipeline.dataset:
            mlflow.log_param('setup_1_period', self.pipeline.dataset['setup_1_period'])

        current_steps['method'].destruct()

    def _finish_experiment(self, best_params: dict) -> dict:
        # Mango uses scikit learn and due to the autolog functionality it logs some runs to the default experiment, so we need to clear the default experiment to avoid confusion
        default_experiment_id = mlflow.get_experiment_by_name("Default").experiment_id

        runs = mlflow.search_runs(experiment_ids=default_experiment_id)

        for run in runs.iterrows():
            run_id = run[1]['run_id']
            mlflow.delete_run(run_id)

        if self.log_best_scores and os.path.exists(self.best_scores_info_dict_path):
            with open(self.best_scores_info_dict_path, 'rb') as file:
                best_scores_info_saved_dict = pickle.load(file)
                best_run_id = best_scores_info_saved_dict['best_run_id']
                pd.DataFrame(best_scores_info_saved_dict['best_scores']).to_csv(f'scores_{best_run_id}.csv',
                                                                                index=False, header=False)

                with mlflow.start_run(run_id=best_run_id, experiment_id=self.experiment_id):
                    mlflow.log_artifact(f'scores_{best_run_id}.csv')
                    os.remove(f'scores_{best_run_id}.csv')

            os.remove(self.best_scores_info_dict_path)

            os.remove(self.lock_file_path)

        return best_params


    def from_time_to_bins(self, labels, n=10):
        bin_size = math.ceil(len(labels) / n)
        sorted_labels = sorted(labels)
        bins = []
        for i in range(n):
            bins.append(sorted_labels[min((i + 1) * bin_size - 1, len(sorted_labels) - 1)])
        return bins

    def mape_mdape_bins(self, preds, labels, n=10):
        bins = self.from_time_to_bins(labels, n)
        bin_dict = {}
        for b in bins:
            bin_dict[b] = {'preds': [], 'labels': []}
        for pred, label in zip(preds, labels):
            for b in bins:
                if label <= b:
                    bin_dict[b]['preds'].append(pred)
                    bin_dict[b]['labels'].append(label)
                    break
        mape_per_bin = []
        mdape_per_bin = []
        for b in bins:
            if len(bin_dict[b]['labels']) == 0:
                continue
            current_mape = mean_absolute_percentage_error([l + 1 for l in bin_dict[b]['labels']],
                                                          [p + 1 for p in bin_dict[b]['preds']])
            current_mdape = self.mdape([l + 1 for l in bin_dict[b]['labels']],
                                       [p + 1 for p in bin_dict[b]['preds']])
            mape_per_bin.append((b, current_mape))
            mdape_per_bin.append((b, current_mdape))
        mape_per_bin = {"time": [bt[0] for bt in mape_per_bin], "MAPE": [br[1] for br in mape_per_bin]}
        mdape_per_bin = {"time": [bt[0] for bt in mdape_per_bin], "MdAPE": [br[1] for br in mdape_per_bin]}
        return mape_per_bin, mdape_per_bin

    def surv_evaluate(self, results_rul, result_scores, result_dates, result_labels, train_labels, plot_dictionary,
                      rtfs):
        # TODO: incorporate for censored data as well, now only for RTF
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        flatten_preds = []
        flatten_labels = []
        is_rtf = []
        for predds, rtff in zip(results_rul, rtfs):
            flatten_preds.extend(predds)
            is_rtf.extend([rtff for i in range(len(predds))])
        for labss in result_labels:
            flatten_labels.extend([labb[0] for labb in labss])

        train_flatten_labels = []
        for labss in train_labels:
            train_flatten_labels.extend([labb[0] for labb in labss])

        flatten_for_rul = [flat_pred for flat_pred, isrtf in zip(flatten_preds, is_rtf) if isrtf == 1]
        labels_for_rul = [flat_pred for flat_pred, isrtf in zip(flatten_labels, is_rtf) if isrtf == 1]

        best_dict = {}
        best_dict['mse'] = mean_squared_error(labels_for_rul, flatten_for_rul)
        best_dict['r2'] = r2_score(labels_for_rul, flatten_for_rul)
        best_dict['mae'] = mean_absolute_error(labels_for_rul, flatten_for_rul)
        best_dict['rmse'] = root_mean_squared_error(labels_for_rul, flatten_for_rul)
        best_dict['mape'] = mean_absolute_percentage_error([l + 1 for l in labels_for_rul],
                                                           [p + 1 for p in flatten_for_rul])
        best_dict['mdape'] = self.mdape([l + 1 for l in labels_for_rul], [p + 1 for p in flatten_for_rul])
        mape_bins, mdape_bins = self.mape_mdape_bins(flatten_for_rul, labels_for_rul, n=10)

        test_preds = []
        for pred in result_scores:
            test_preds.extend([inpred[0] for inpred in pred])

        times = np.unique([ty for ty in train_flatten_labels])
        times.sort()
        test_y = [(rtf, ty) for ty, rtf in zip(flatten_labels, is_rtf)]

        eval_survs = self.surv_eval(test_y, test_preds, times=times, train_y=None)
        eval_survs['mape_bins'] = mape_bins
        eval_survs['mdape_bins'] = mdape_bins
        if self.debug:
            self._plot_SA(plot_dictionary)

        aditional_data = {}
        for key in eval_survs:
            if key in ['brier_scores', 'roc_auc_list', 'c_index_list', "mape_bins", "mdape_bins", "IBR_bins"]:
                aditional_data[key] = eval_survs[key]
                mlflow.log_table(aditional_data[key], f"survival_{key}.json")
            else:
                best_dict[key] = eval_survs[key]

        # MdAPE stored in  best_dict['mdape']
        # IBS stored in best_dict['IBS']
        inverted_mdape = 1 - min(1, best_dict['mdape'])
        inverted_ibs = 1 - 4 * min(0.25, best_dict['IBS'])
        beta = 1
        best_dict['CombinedScore'] = (1 + beta ** 2) * inverted_mdape * inverted_ibs / (
                    beta ** 2 * inverted_mdape + inverted_ibs)
        mlflow.log_metrics(best_dict)


        mlflow.log_metrics(best_dict)

        return best_dict

    def surv_eval(self, test_y, test_preds, times=None, train_y=None):
        """
        test_y: structured array with (event, time) shape: (nsamples)
        test_preds: array of shape: (nsamples, ntimes) with predicted survival probabilities
        times: list of time points at which predictions are made shape: (ntimes)
        """
        from sksurv.metrics import brier_score, integrated_brier_score, cumulative_dynamic_auc
        from sksurv.metrics import concordance_index_censored

        test_preds = np.array(test_preds)
        if train_y is None:
            train_y = np.array(test_y, dtype=[('event', 'bool'), ('time', 'float')])
            # and keep only fatal
            new_test_y = []
            new_test_preds = []
            for ty, preds_i in zip(test_y, test_preds):
                new_test_y.append(ty)
                new_test_preds.append(preds_i)
            test_y = new_test_y
            test_y = np.array(test_y, dtype=[('event', 'bool'), ('time', 'float')])
            test_preds = np.array(new_test_preds)
        if times is None:
            times = np.unique([ty[1] for ty in train_y])
            times.sort()

        maxtt = max([ty[1] for ty in test_y])
        mintt = min([ty[1] for ty in test_y])
        pos = 0
        for i, t in enumerate(times):
            if t >= maxtt:
                break
            pos = i
        pos_pre = 0
        for i, t in enumerate(times):
            if t >= mintt:
                pos_pre = i
                break

        times = times[pos_pre:pos]
        test_preds = test_preds[:, pos_pre:pos]

        b_times, b_score = brier_score(train_y, test_y, test_preds, times)
        integrated_brier_score_value = integrated_brier_score(train_y, test_y, test_preds, times)

        roc_pt, mean_roc = cumulative_dynamic_auc(train_y, test_y, -test_preds, times)

        evals = {}
        # do it in 20 time points evenly spaced
        target_times = np.linspace(times.min(), times.max(), 20)
        positions = [np.argmin(np.abs(times - t)) for t in target_times]
        cis = []
        events = [ty[0] for ty in test_y]
        times_for_c = [ty[1] for ty in test_y]

        # res = concordance_index_censored(events, times_for_c, [max_rul - pred_rul for pred_rul in flatten_preds])
        for i in positions:
            res = concordance_index_censored(events, times_for_c, [1 - tpred for tpred in test_preds[:, i]])
            cis.append((times_for_c[i], res[0]))
        summaris = [np.sum(test_preds[i, :]) for i in range(len(test_preds))]
        maxsum = max(summaris)
        sumres = concordance_index_censored(events, times_for_c, [maxsum - summm for summm in summaris])
        cis.append((-1, sumres[0]))  # add last time point again
        # mean_ic = np.mean([ci[1] for ci in cis])
        evals['brier_scores'] = {"time": [bt for bt in b_times], "Brier": [br for br in b_score]}
        evals['roc_auc_list'] = {"time": [bt for bt in times], "ROC": [br for br in roc_pt]}
        evals['c_index_list'] = {"time": [ci[0] for ci in cis], "C-Index": [ci[1] for ci in cis]}
        evals['c_index_mean'] = np.mean([ci[1] for ci in cis])
        evals['c_index'] = np.max([ci[1] for ci in cis])
        evals['IBS'] = integrated_brier_score_value
        evals['Max_brier'] = np.max(b_score)
        evals['mean_roc'] = mean_roc
        # evals['IBR_bins']=self.IBR_bins(train_y, test_y, test_preds, times, n=10)

        return evals

    def IBR_bins(self, train_y, test_y, test_preds, times, n=10):
        from sksurv.metrics import integrated_brier_score
        bins = self.from_time_to_bins([ty[1] for ty in test_y], n)
        bin_dict = {}
        for b in bins:
            bin_dict[b] = {'train_y': [], 'test_y': [], 'test_preds': []}
        for ty, preds_i in zip(test_y, test_preds):
            pre_bin = 0
            for b in bins:
                if ty[1] <= b:
                    bin_dict[b]['test_y'].append(ty)
                    bin_dict[b]['test_preds'].append(
                        [pred for ts, pred in zip(times, preds_i) if ts < b and ts > pre_bin])
                    break
                pre_bin = b
        for ty in train_y:
            for b in bins:
                if ty[1] <= b:
                    bin_dict[b]['train_y'].append(ty)
                    break
        ibr_per_bin = []
        pre_bin = 0
        for b in bins:
            if len(bin_dict[b]['test_y']) == 0:
                continue
            current_ibr = integrated_brier_score(
                np.array(bin_dict[b]['train_y'], dtype=[('event', 'bool'), ('time', 'float')]),
                np.array(bin_dict[b]['test_y'], dtype=[('event', 'bool'), ('time', 'float')]),
                np.array(bin_dict[b]['test_preds']),
                [tt for tt in times if tt < b and tt > pre_bin])
            pre_bin = b
            ibr_per_bin.append((b, current_ibr))
        ibr_per_bin = {"time": [bt[0] for bt in ibr_per_bin], "IBR": [br[1] for br in ibr_per_bin]}
        return ibr_per_bin

    def mdape(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        ape = np.abs((y_true - y_pred) / y_true)
        return np.median(ape)

    def rul_evaluate(self, result_scores, result_dates, result_labels, plot_dictionary, rtfs):
        """

        :param result_scores: a list of lists with predictions
        :param result_dates: a list of lists with the dates of the predictions
        :param results_isfailure:
        :param plot_dictionary:
        :return:
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        flatten_preds = []
        flatten_labels = []
        is_rtf = []
        for predds, rtff in zip(result_scores, rtfs):
            flatten_preds.extend(predds)
            is_rtf.extend([rtff for i in range(len(predds))])
        for labss in result_labels:
            flatten_labels.extend(labss)

        flatten_for_rul = [flat_pred for flat_pred, isrtf in zip(flatten_preds, is_rtf) if isrtf == 1]
        labels_for_rul = [flat_pred for flat_pred, isrtf in zip(flatten_labels, is_rtf) if isrtf == 1]

        best_dict = {}
        best_dict['mse'] = mean_squared_error(labels_for_rul, flatten_for_rul)
        best_dict['r2'] = r2_score(labels_for_rul, flatten_for_rul)
        best_dict['mae'] = mean_absolute_error(labels_for_rul, flatten_for_rul)
        best_dict['rmse'] = root_mean_squared_error(labels_for_rul, flatten_for_rul)
        best_dict['mape'] = mean_absolute_percentage_error([l + 1 for l in labels_for_rul],
                                                           [p + 1 for p in flatten_for_rul])
        best_dict['mdape'] = self.mdape([l + 1 for l in labels_for_rul], [p + 1 for p in flatten_for_rul])
        mape_bins, mdape_bins = self.mape_mdape_bins(flatten_for_rul, labels_for_rul, n=10)

        # transorm to survival predictions and evaluate survival
        times = np.unique([ty for ty in flatten_labels])
        times.sort()
        test_y = [(rtf, ty) for ty, rtf in zip(flatten_labels, is_rtf)]

        eval_survs, max_rul = self.surv_eval_for_rul(test_y, flatten_preds, result_scores, result_labels, rtfs,
                                                     times=times, train_y=None)
        eval_survs['mape_bins'] = mape_bins
        eval_survs['mdape_bins'] = mdape_bins

        aditional_data = {}
        for key in eval_survs:
            if key in ['brier_scores', 'roc_auc_list', 'c_index_list', "mape_bins", "mdape_bins", "IBR_bins"]:
                aditional_data[key] = eval_survs[key]
                mlflow.log_table(aditional_data[key], f"survival_{key}.json")
            else:
                best_dict[key] = eval_survs[key]

        # MdAPE stored in  best_dict['mdape']
        # IBS stored in best_dict['IBS']
        inverted_mdape=1-min(1, best_dict['mdape'])
        inverted_ibs=1-4*min(0.25, best_dict['IBS'])
        beta=1
        best_dict['CombinedScore'] = (1+beta**2)*inverted_mdape*inverted_ibs/(beta**2*inverted_mdape+inverted_ibs)
        mlflow.log_metrics(best_dict)

        return best_dict

    # def surv_eval(self,result_scores, result_dates,result_labels, plot_dictionary):
    def surv_eval_for_rul(self, test_y, flatten_preds, result_scores, result_labels, is_failure, times=None,
                          train_y=None):

        """
        test_y: structured array with (event, time) shape: (nsamples)
        test_preds: array of shape: (nsamples, ntimes) with predicted survival probabilities
        times: list of time points at which predictions are made shape: (ntimes)
        """
        from sksurv.metrics import brier_score, integrated_brier_score, cumulative_dynamic_auc
        from sksurv.metrics import concordance_index_censored

        new_test_y = []
        new_flatten_preds = []
        for ty, preds_i in zip(test_y, flatten_preds):
            new_test_y.append(ty)
            new_flatten_preds.append(preds_i)
        test_y = new_test_y
        test_y = np.array(test_y, dtype=[('event', 'bool'), ('time', 'float')])

        flatten_preds = np.array(new_flatten_preds)

        if train_y is None:
            train_y = np.array(test_y, dtype=[('event', 'bool'), ('time', 'float')])
            # and keep only fatal

        if times is None:
            times = np.unique([ty[1] for ty in train_y])
            times.sort()

        maxtt = max([ty[1] for ty in test_y])
        mintt = min([ty[1] for ty in test_y])
        pos = 0
        for i, t in enumerate(times):
            if t >= maxtt:
                break
            pos = i
        pos_pre = 0
        for i, t in enumerate(times):
            if t >= mintt:
                pos_pre = i
                break

        times = times[pos_pre:pos]

        evals = {}

        # C-index
        events = [ty[0] for ty in test_y]
        times_for_c = [ty[1] for ty in test_y]
        max_rul = max(flatten_preds)
        res = concordance_index_censored(events, times_for_c, [max_rul - pred_rul for pred_rul in flatten_preds])
        evals['c_index'] = res[0]

        test_preds = hard_transform_survival(times, flatten_preds)

        plot_test_preds = []
        for pred_set in result_scores:
            plot_test_preds.append([[predss, times] for predss in sigmoid_survival_batch(times, pred_set, tau=max(times)/100)])
        self.plot_SA_of_RUL(plot_test_preds, result_labels, is_failure)

        b_times, b_score = brier_score(train_y, test_y, test_preds, times)
        evals['brier_scores'] = {"time": [bt for bt in b_times], "Brier": [br for br in b_score]}
        evals['Max_brier_HM'] = np.max(b_score)

        roc_pt, mean_roc = cumulative_dynamic_auc(train_y, test_y, -test_preds, times)
        evals['roc_auc_list'] = {"time": [bt for bt in times], "ROC": [br for br in roc_pt]}
        evals['mean_roc'] = mean_roc

        integrated_brier_score_value = integrated_brier_score(train_y, test_y, test_preds, times)
        evals['IBS_HM'] = integrated_brier_score_value

        test_preds = sigmoid_survival_batch(times, flatten_preds, tau=max(times)/100)
        integrated_brier_score_value = integrated_brier_score(train_y, test_y, test_preds, times)
        evals['IBS'] = integrated_brier_score_value
        b_times, b_score = brier_score(train_y, test_y, test_preds, times)
        evals['Max_brier'] = np.max(b_score)
        # evals["IBR_bins"]=self.IBR_bins(train_y, test_y, test_preds, times, n=10)

        return evals, max_rul

    def _check_cached_run(self, params: dict):
        current_params = params.copy()

        if 'profile_size' in current_params:
            current_params['auto_flavor_profile_size'] = current_params['profile_size']
            del current_params['profile_size']

        method_params = {re.sub('method_', '', k): v for k, v in current_params.items() if 'method' in k}
        preprocessor_params = {re.sub('preprocessor_', '', k): v for k, v in current_params.items() if
                               'preprocessor' in k}
        postprocessor_params = {re.sub('postprocessor_', '', k): v for k, v in current_params.items() if
                                'postprocessor' in k}
        thresholder_params = {re.sub('thresholder_', '', k): v for k, v in current_params.items() if 'thresholder' in k}

        runs = mlflow.search_runs(self.experiment_id, filter_string='attributes.status = "FINISHED"')

        found_match, found_index, found_run = False, -1, None
        for index, current_run in runs.iterrows():
            found_match = True
            found_index = index
            found_run = current_run

            for param_name, param_value in current_params.items():
                if 'params.' + param_name not in current_run.index:
                    found_match = False
                    break

                if current_run.loc['params.' + param_name] != str(param_value):
                    found_match = False
                    break

            current_steps = {
                'method': self.pipeline.method(event_preferences=self.pipeline.event_preferences, **method_params),
                'preprocessor': self.pipeline.preprocessor(event_preferences=self.pipeline.event_preferences,
                                                           **preprocessor_params),
                'postprocessor': self.pipeline.postprocessor(event_preferences=self.pipeline.event_preferences,
                                                             **postprocessor_params),
                'thresholder': self.pipeline.thresholder(event_preferences=self.pipeline.event_preferences,
                                                         **thresholder_params)
            }

            for step in self.pipeline.get_steps().keys():
                if current_run.loc['params.' + step] != str(current_steps[step]):
                    found_match = False
                    break

            predictive_horizon_to_check, beta_to_check, lead_to_check = -1, -1, -1
            if "anomaly_ranges" in self.pipeline.dataset.keys():
                if self.pipeline.dataset["anomaly_ranges"]:
                    predictive_horizon_to_check = self.pipeline.slide
                    beta_to_check = self.pipeline.beta
                    lead_to_check = self.pipeline.slide
                else:
                    predictive_horizon_to_check = self.pipeline.predictive_horizon
                    beta_to_check = self.pipeline.beta
                    lead_to_check = self.pipeline.lead
            else:
                predictive_horizon_to_check = self.pipeline.predictive_horizon
                beta_to_check = self.pipeline.beta
                lead_to_check = self.pipeline.lead

            if str(predictive_horizon_to_check) != current_run.loc['params.predictive_horizon'] \
                    or str(beta_to_check) != current_run.loc['params.beta'] \
                    or str(lead_to_check) != current_run.loc['params.lead']:
                found_match = False

            if found_match:
                break

        if found_match:
            logging.info(
                f'Found cached run with parameters: {current_params}, steps={[str(step) for step in current_steps.values()]}, predictive_horizon={predictive_horizon_to_check}, beta={beta_to_check} and lead={lead_to_check}. Skipping...')
            return found_run.loc['metrics.' + self.optimization_param]
        else:
            return None


def root_mean_squared_error(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))