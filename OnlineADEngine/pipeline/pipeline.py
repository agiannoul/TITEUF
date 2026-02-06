from typing import TypedDict, List
import pandas as pd
from OnlineADEngine.preprocessing.record_level.default import DefaultPreProcessor
from OnlineADEngine.postprocessing.default import DefaultPostProcessor
from OnlineADEngine.thresholding.constant import ConstantThresholder
from OnlineADEngine.thresholding.thresholder import ThresholderInterface
from OnlineADEngine.method.method import MethodInterface
from OnlineADEngine.postprocessing.post_processor import PostProcessorInterface
from OnlineADEngine.utils.utils import expand_event_preferences


class PdMPipelineSteps(TypedDict):
    preprocessor:  DefaultPreProcessor
    method : MethodInterface
    postprocessor : PostProcessorInterface
    thresholder : ThresholderInterface


class PdMPipeline():
    def __init__(self,
                steps: PdMPipelineSteps,
                dataset: dict,
                auc_resolution : int,
                experiment_type,
    ):
        self.dataset = dataset
        self.steps = steps
        self.event_data = dataset['event_data']
        self.event_data['date']=pd.to_datetime( self.event_data['date'])
        self.event_preferences = dataset['event_preferences']
        self.target_dates = dataset['dates']
        self.historic_dates = dataset['dates']
        self.predictive_horizon = dataset['predictive_horizon']
        self.slide = dataset['slide']
        self.lead = dataset['lead']
        self.beta = dataset['beta']
        self.auc_resolution = auc_resolution

        self.preprocessor = steps.get('preprocessor', DefaultPreProcessor(event_preferences=self.event_preferences))
        self.method = steps.get('method', experiment_type)
        self.postprocessor = steps.get('postprocessor', DefaultPostProcessor(event_preferences=self.event_preferences))

        self.thresholder = steps.get('thresholder', ConstantThresholder(threshold_value=0.5, event_preferences=self.event_preferences))


    def get_steps(self) -> PdMPipelineSteps:
        return self.steps


    def get_step_by_name(self, step_name: str):
        return self.steps[step_name]

    def extract_failure_dates_for_source(self, source: str) -> list[pd.Timestamp]:
        result = []
        try:
            expanded_event_preferences = self.expanded_event_preferences
            source_event_dict = self.source_event_dict
            get_affected_failure = self.get_affected_failure
            get_affected_reset = self.get_affected_reset
        except Exception as e:
            self.expanded_event_preferences = expand_event_preferences(event_data=self.event_data,
                                                                       event_preferences=self.event_preferences)
            expanded_event_preferences = self.expanded_event_preferences
            self.source_event_dict = {}
            get_affected_failure = {}
            get_affected_reset = {}
            for row in self.event_data.itertuples():
                if row.source not in self.source_event_dict:
                    self.source_event_dict[row.source] = []
                    get_affected_failure[row.source] = []
                    get_affected_reset[row.source] = []
                self.source_event_dict[row.source].append(row)
            for key in self.source_event_dict:
                self.source_event_dict[key] = pd.DataFrame(self.source_event_dict[key])

            get_affected_failure = self.find_affected_sources(expanded_event_preferences['failure'],
                                                              get_affected_failure)
            get_affected_reset = self.find_affected_sources(expanded_event_preferences['reset'], get_affected_reset)

            self.get_affected_failure = get_affected_failure
            self.get_affected_reset = get_affected_reset
            source_event_dict = self.source_event_dict

        for current_pref_ in get_affected_failure[source]:
            for row_index, row in source_event_dict[current_pref_[0]].iterrows():
                if row['type'] == current_pref_[2] and row['description'] == current_pref_[1]:
                    result.append(row['date'])

        return sorted(list(set(result)))

    def find_affected_sources(self, given_expanded_preferences, get_affected) -> dict[str, List[List[str]]]:
        for current_preference in given_expanded_preferences:
            if current_preference.target_sources == '=':
                get_affected[current_preference.source].append(
                    [current_preference.source, current_preference.description, current_preference.type])
            elif current_preference.target_sources == '*':
                for source_key in self.source_event_dict.keys():
                    get_affected[source_key].append(
                        [current_preference.source, current_preference.description, current_preference.type])
            else:
                for target_source in current_preference.target_sources:
                    get_affected[target_source].append(
                        [current_preference.source, current_preference.description, current_preference.type])
        return get_affected

    def extract_reset_dates_for_source(self, source) -> list[pd.Timestamp]:
        result = []
        try:
            expanded_event_preferences = self.expanded_event_preferences
            source_event_dict = self.source_event_dict
            get_affected_failure = self.get_affected_failure
            get_affected_reset = self.get_affected_reset
        except Exception as e:
            self.expanded_event_preferences = expand_event_preferences(event_data=self.event_data,
                                                                       event_preferences=self.event_preferences)
            expanded_event_preferences = self.expanded_event_preferences
            self.source_event_dict = {}
            get_affected_failure = {}
            get_affected_reset = {}
            for row in self.event_data.itertuples():
                if row.source not in self.source_event_dict:
                    self.source_event_dict[row.source] = []
                    get_affected_failure[row.source] = []
                    get_affected_reset[row.source] = []
                self.source_event_dict[row.source].append(row)
            for key in self.source_event_dict:
                self.source_event_dict[key] = pd.DataFrame(self.source_event_dict[key])

            get_affected_failure = self.find_affected_sources(expanded_event_preferences['failure'],
                                                              get_affected_failure)
            get_affected_reset = self.find_affected_sources(expanded_event_preferences['reset'], get_affected_reset)

            self.get_affected_failure = get_affected_failure
            self.get_affected_reset = get_affected_reset
            source_event_dict = self.source_event_dict

        for current_pref_ in get_affected_reset[source]:
            for row_index, row in source_event_dict[current_pref_[0]].iterrows():
                if row['type'] == current_pref_[2] and row['description'] == current_pref_[1]:
                    result.append(row['date'])

        return sorted(list(set(result)))


    def get_steps_as_str(self):
        return f'preprocessor_{self.steps["preprocessor"]}_method_{self.steps["method"]}_postprocessor_{self.steps["postprocessor"]}_thresholder_{self.steps["thresholder"]}'