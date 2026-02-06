import random

import pandas as pd


class Dataset:
    """
    A class to handle dataset preparation and processing for predictive maintenance tasks.
    This includes splitting data into episodes, calculating sliding windows, and preparing
    training, validation, and testing datasets.

    Parameters
        ----------
        data : pd.DataFrame
            The input data containing time-series information. If `event_df` is not provided but `maintenance_column` and `failure_column` are provided
            it except that `maintenance_column` and `failure_column` are included in data, and use them to derive on the episodes.
            If `maintenance_column` and `failure_column` and `event_df` are not provided it assumes every source as a single run-to-failure episode.
        datetime_column : str
            The name of the column representing datetime values.
        event_indicator : str, default=None
            The name of the column indicating event occurrence (binary). If provided, it is used to derive episode ending (0: maintenance/reset or 1: failure).
        source_column : str, default='source'
            The name of the column representing the source of the data.
        beta : int, default=1
            A parameter used for objective calculations.
        slide : int, optional
            The sliding window size. If None, it is calculated automatically.
        lead : str, default="2 seconds"
            The lead time for predictions.
        predictive_horizon : str, optional
            The predictive horizon for the dataset. If None, it is calculated automatically.
        train_sources : float or list, default=0.6
            The ratio (float) or list of source names used for training. If a float, it represents the proportion of sources used for training.
        val_sources : float or list, default=0.2
            The ratio (float) or list of source names used for validation. If a float, it represents the proportion of sources used for validation.
        test_sources : float or list, default=0.2
            The ratio (float) or list of source names used for testing. If a float, it represents the proportion of sources used for testing.
        in_source_split : bool, default=False Whether to select train/val/test sources from within each source (True) or from the overall sources (False).
    """
    def __init__(self,data,datetime_column,event_indicator="event",source_column='source',
                 beta=1, slide=None, lead="0 seconds",predictive_horizon=None,
                 train_sources=0.6,val_sources=0.2,test_sources=0.2,in_source_split=False,DIVIDER=3600):

        # Dataset
        self.in_source_split=in_source_split
        self.datetime_column = datetime_column
        data[self.datetime_column]=pd.to_datetime(data[self.datetime_column])
        data[source_column]=data[source_column].astype(str)
        episodes, run_to_failure,_,original_s_has_f = episodes_formulation(data, datetime_column,event_indicator, None,
                                                                None,None, source_column,DIVIDER=DIVIDER)
        self.original_sources=data[source_column].unique().tolist()
        self.sources = [ep.iloc[0][source_column] for ep in episodes]
        self.original_s_has_f =original_s_has_f


        self.source_column = source_column

        self.train_sources = train_sources
        self.val_sources = val_sources
        self.test_sources = test_sources

        self.train_source_name='train'
        self.split_sources_to_train_test_val(episodes,run_to_failure)
        # Calcuylated in split_sources_to_train_test_val:
        # self.matches = matches
        # self.rtf_dict
        # self.train_dfs = train_dfs
        # self.val_dfs = val_dfs
        # self.test_dfs = test_dfs
        #
        # self.sources_for_train = for_train
        # self.sources_for_val = for_val
        # self.sources_for_test = for_test

        self.max_wait_time = max(10,int(2*min([ep.shape[0] for ep in episodes])/3))


        self.rul_column = 'RUL'


        # Objective parameters
        self.beta = beta
        self.lead = lead
        # when lead and predictive_horizon are not provided, set them to default values
        if predictive_horizon is None:
            durations=[(ep.iloc[-1][self.datetime_column]-ep.iloc[0][self.datetime_column]).total_seconds()/3600.0 for ep,rtf in zip(episodes,run_to_failure) if rtf==1]
            self.predictive_horizon = f"{min(durations)/10.0} hours"
        else:
            self.predictive_horizon = predictive_horizon
        if slide is None:
            self.slide = self.slide_calculation(episodes, run_to_failure)
        else:
            self.slide = slide

    def slide_calculation(self,episodes, run_to_failure):
        """
        Slide + Predictive Horizon has to 1/3 of the smallest episode in default
        Returns
        -------

        """
        minlen = float('inf')
        minep = None
        for ep, rtfi in zip(episodes, run_to_failure):
            if rtfi == 1:
                dur = ( ep.iloc[-1][self.datetime_column]-ep.iloc[0][self.datetime_column]).total_seconds() / 3600.0
                if dur < minlen:
                    minlen = dur
                    minep = ep
        length_min_ep = minep.shape[0]
        lastime = minep.iloc[-1][self.datetime_column]
        pos = 0
        for i in range(length_min_ep):
            ctime = minep.iloc[i][self.datetime_column]
            if ctime >= lastime - pd.Timedelta(self.predictive_horizon):
                pos = i
                break
        ph_length = length_min_ep - pos
        slide=int(length_min_ep / 3) - ph_length
        return max(slide,1)

    def split_sources_to_train_test_val(self,episodes,ran_to_failure):
        """
        Splits the sources into training, validation, and testing datasets.

        Parameters
        ----------
        episodes : list
            A list of dataframes, where each dataframe corresponds to an episode.
        ran_to_failure : list
            A list of integers indicating whether each episode is a run-to-failure (1) or not (0).

        Returns
        -------
        None
            The method updates the following attributes of the class:
            - self.train_dfs: Dataframes for training.
            - self.val_dfs: Dataframes for validation.
            - self.test_dfs: Dataframes for testing.
            - self.sources_for_train: Sources used for training.
            - self.sources_for_val: Sources used for validation.
            - self.sources_for_test: Sources used for testing.
            - self.matches: A dictionary mapping training sources to validation and testing sources.
        """
        for i in range(len(episodes)):
            episodes[i][self.datetime_column]=pd.to_datetime(episodes[i][self.datetime_column])
        unvid=self.sources
        self.rtf_dict={source: rtf for source, rtf in zip(unvid, ran_to_failure)}

        for_train = []
        for_val = []
        for_test = []
        if isinstance(self.train_sources, float) and isinstance(self.val_sources, float) and isinstance(self.test_sources, float):
            if self.train_sources + self.val_sources + self.test_sources != 1.0:
                raise ValueError("When train_sources, val_sources, test_sources  are pass as floats (ratio), they must sum 1.")

            for_train = []
            # if there are at least three ORIGINAL SOURCES that contain failure
            if self.in_source_split==False and len([1 for key in self.original_sources if  self.original_s_has_f[key]])>=3:
                random.seed(42)
                or_sources_with_failure = [sui for sui in self.original_sources if self.original_s_has_f[sui]]
                or_sources_without_failure = list(set(self.original_sources) - set(or_sources_with_failure))

                train_source, val_source, test_source = self.safe_splitting(or_sources_with_failure)
                c_train_source, c_val_source, c_test_source = self.safe_splitting(or_sources_without_failure)

                self.train_sources = train_source + c_train_source
                self.val_sources = val_source + c_val_source
                self.test_sources = test_source + c_test_source


                for source in unvid:
                    if source.split("_ep")[0] in self.val_sources:
                        for_val.append(source)
                    elif source.split("_ep")[0] in self.test_sources:
                        for_test.append(source)
                    elif source.split("_ep")[0] in self.train_sources:
                        for_train.append(source)
            # Either in_source_split is True or there are less than three original sources with failure
            # look at episode level spliting
            else:
                self.train_sources = []
                self.val_sources = []
                self.test_sources = []
                at_least_one_failure_in_train=True
                for orginal_source in self.original_sources:
                    source_episodes = [ep for ep in episodes if
                                       ep[self.source_column].iloc[0].startswith(orginal_source)]
                    train_source, val_source, test_source = self.safe_splitting(source_episodes,at_least_one_failure_in_train)
                    at_least_one_failure_in_train=False
                    self.train_sources.extend(train_source)
                    self.val_sources.extend(val_source)
                    self.test_sources.extend(test_source)
                    for_train.extend(train_source)
                    for_val.extend(val_source)
                    for_test.extend(test_source)
        else:
            for_train.extend(self.train_sources)
            for_val.extend(self.val_sources)
            for_test.extend(self.test_sources)


        if self.train_sources is None or self.val_sources is None or self.test_sources is None:
            raise ValueError(
                "Either provide train_sources,val_sources and test_sources as ratio or as list of sources.")
        else:

            traif = max([self.rtf_dict[source] for source in for_train])
            testf = max([self.rtf_dict[source] for source in for_test])
            valf = max([self.rtf_dict[source] for source in for_val])
            if traif+testf+valf<3:
                raise ValueError("At least one source/episode with failure event must be present in each of train, val and test sets.")


            train_dfs = [episodes[i] for i, ep in enumerate(episodes) if ep[self.source_column].iloc[0] in for_train]
            val_dfs = [episodes[i] for i, ep in enumerate(episodes) if ep[self.source_column].iloc[0] in for_val]
            test_dfs = [episodes[i] for i, ep in enumerate(episodes) if ep[self.source_column].iloc[0] in for_test]

            matches = {}
            for source in for_val + for_test:
                matches[source] = self.train_source_name
        self.matches=matches

        self.train_dfs=train_dfs
        self.val_dfs=val_dfs
        self.test_dfs=test_dfs

        self.sources_for_train=for_train
        self.sources_for_val = for_val
        self.sources_for_test=for_test

    def safe_splitting(self,source_episodes,at_least_one_failure_in_train=False):
        train_source = []
        val_source = []
        test_source = []
        train_source_count = 0
        val_source_count = 0
        test_source_count = 0
        if len(source_episodes) >= 3:
            val_source_count = int(self.val_sources * len(source_episodes))
            test_source_count = int(self.test_sources * len(source_episodes))
            train_source_count = len(source_episodes) - val_source_count - test_source_count
            if test_source_count == 0:
                test_source_count = 1
                train_source_count -= 1
            if val_source_count == 0:
                val_source_count = 1
                train_source_count -= 1
        elif len(source_episodes) == 2:
            test_source_count = 1
            val_source_count = 1
        else:
            test_source_count = 1
        fail_in_train=False
        for i, source in enumerate(source_episodes):
            if i < train_source_count and (at_least_one_failure_in_train==False or fail_in_train):
                if self.rtf_dict.get(source,0)==1:
                    fail_in_train=True
                train_source.append(source)
            elif i < train_source_count + val_source_count:
                val_source.append(source)
            else:
                test_source.append(source)
        return train_source, val_source, test_source
    def get_rul_dataset(self,keep_sources=None):
        concatinated_train = pd.concat([df for df in self.train_dfs if self.rtf_dict[df.iloc[0][self.source_column]]==1], ignore_index=True)


        cols_to_drop = [self.source_column, self.rul_column]

        if keep_sources is not None and keep_sources in cols_to_drop:
            cols_to_drop.remove(keep_sources)

        dataset = {}
        dataset['match_sources'] = self.matches
        dataset['target_sources'] = [str(vid) for vid in self.sources_for_val]

        target_data=[]
        for df in self.val_dfs:
            tdf=df.copy()
            tdf=tdf.drop(columns=cols_to_drop).reset_index(drop=True).copy()
            if keep_sources is not None:
                tdf[keep_sources]=df[self.source_column]
            target_data.append(tdf)
        dataset['target_data'] = target_data
        dataset['is_failure'] = [self.rtf_dict[str(vid)] for vid in self.sources_for_val]
        dataset['target_labels'] = [df[self.rul_column].values for df in self.val_dfs]


        if keep_sources is not None:
            concatinated_train[keep_sources]=[s for s in concatinated_train[self.source_column]]
        dataset['historic_data'] = [concatinated_train.drop(columns=cols_to_drop)]
        dataset['historic_sources'] = [self.train_source_name]
        dataset['anomaly_labels'] = [concatinated_train[self.rul_column].values]
        dataset["dates"] = self.datetime_column

        from OnlineADEngine.pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple

        event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

        event_preferences: EventPreferences = {
            'failure': [],
            'reset': []
        }
        dataset["event_preferences"] = event_preferences
        dataset["event_data"] = event_data
        dataset['predictive_horizon'] = self.predictive_horizon
        dataset['slide'] = self.slide
        dataset['lead'] = self.lead
        dataset['beta'] = self.beta

        ############## test dataset ###############

        test_dataset = {}
        test_dataset['match_sources'] = self.matches
        test_dataset['target_sources'] = [str(vid) for vid in self.sources_for_test]

        target_data = []
        for df in self.test_dfs:
            tdf = df.copy()
            tdf= tdf.drop(columns=cols_to_drop).reset_index(drop=True).copy()
            if keep_sources is not None:
                tdf[keep_sources] = df[self.source_column]
            target_data.append(tdf)
        test_dataset['target_data'] = target_data
        test_dataset['target_labels'] = [df[self.rul_column].values for df in self.test_dfs]
        test_dataset['is_failure'] = [self.rtf_dict[str(vid)] for vid in self.sources_for_test]


        if keep_sources is not None:
            concatinated_train[keep_sources] = [s for s in concatinated_train[self.source_column]]
        test_dataset['historic_data'] = [concatinated_train.drop(columns=cols_to_drop)]
        test_dataset['historic_sources'] = [self.train_source_name]
        test_dataset['anomaly_labels'] = [concatinated_train[self.rul_column].values]

        test_dataset["dates"] = self.datetime_column

        test_dataset["event_preferences"] = event_preferences
        test_dataset["event_data"] = event_data
        test_dataset['predictive_horizon'] = self.predictive_horizon
        test_dataset['slide'] = self.slide
        test_dataset['lead'] = self.lead
        test_dataset['beta'] = self.beta

        return dataset, test_dataset

    def df_to_x_y_surv(self,df,indicator=None):
        if indicator is None:
            y = [(rul, ev) for ev, rul in zip(df["event"], df[self.rul_column])]
        else:
            y = [(rul, indicator) for rul in df[self.rul_column]]
        return y

    def get_SA_dataset(self,keep_sources=None):
        train_dfs_with_events=[]
        for df in self.train_dfs:
            df_with_event=df.copy()
            df_with_event["event"]=self.rtf_dict[df.iloc[0][self.source_column]]
            train_dfs_with_events.append(df_with_event)
        concatinated_train = pd.concat(train_dfs_with_events, ignore_index=True)

        #TO-DO: investigate how to deal with case of only run to failure episodes
        if concatinated_train["event"].min()>0:
            event_list=[ev for ev in concatinated_train["event"]]
            event_list[0]=0
            concatinated_train["event"]=event_list


        cols_to_drop = ["event", self.source_column, self.rul_column]
        if keep_sources is not None and keep_sources in cols_to_drop:
            cols_to_drop.remove(keep_sources)
        dataset = {}
        dataset['match_sources'] = self.matches
        dataset['target_sources'] = [str(vid) for vid in self.sources_for_val]

        target_data = []
        for df in self.val_dfs:
            tdf = df.copy()
            tdf=tdf.drop(columns=[self.source_column, self.rul_column]).reset_index(drop=True).copy()
            if keep_sources is not None:
                tdf[keep_sources] = df[self.source_column]
            target_data.append(tdf)
        dataset['target_data'] = target_data
        dataset['target_labels'] = [self.df_to_x_y_surv(df,indicator=self.rtf_dict[df.iloc[0][self.source_column]]) for df in self.val_dfs]
        dataset['is_failure'] = [self.rtf_dict[str(vid)] for vid in self.sources_for_val]
        if keep_sources is not None:
            concatinated_train[keep_sources] = [s for s in concatinated_train[self.source_column]]
        dataset['historic_data'] = [concatinated_train.drop(columns=cols_to_drop)]
        dataset['historic_sources'] = [self.train_source_name]
        dataset['anomaly_labels'] = [self.df_to_x_y_surv(concatinated_train)]

        dataset["dates"] = self.datetime_column

        from OnlineADEngine.pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple

        event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

        event_preferences: EventPreferences = {
            'failure': [],
            'reset': []
        }
        dataset["event_preferences"] = event_preferences
        dataset["event_data"] = event_data
        dataset['predictive_horizon'] = self.predictive_horizon
        dataset['slide'] = self.slide
        dataset['lead'] = self.lead
        dataset['beta'] = self.beta

        ############## test dataset ###############

        test_dataset = {}
        test_dataset['match_sources'] = self.matches
        test_dataset['target_sources'] = [str(vid) for vid in self.sources_for_test]

        target_data = []
        for df in self.test_dfs:
            tdf = df.copy()
            tdf = tdf.drop(columns=[self.source_column, self.rul_column]).reset_index(drop=True).copy()
            if keep_sources is not None:
                tdf[keep_sources] = df[self.source_column]
            target_data.append(tdf)
        test_dataset['target_data'] = target_data
        test_dataset['target_labels'] = [self.df_to_x_y_surv(df, indicator=self.rtf_dict[df.iloc[0][self.source_column]]) for
                                    df in self.test_dfs]
        test_dataset['is_failure'] = [self.rtf_dict[str(vid)] for vid in self.sources_for_test]
        if keep_sources is not None:
            concatinated_train[keep_sources] = [s for s in concatinated_train[self.source_column]]
        test_dataset['historic_data'] = [concatinated_train.drop(columns=cols_to_drop)]
        test_dataset['historic_sources'] = [self.train_source_name]
        test_dataset['anomaly_labels'] = [self.df_to_x_y_surv(concatinated_train)]

        test_dataset["dates"] = self.datetime_column

        test_dataset["event_preferences"] = event_preferences
        test_dataset["event_data"] = event_data
        test_dataset['predictive_horizon'] = self.predictive_horizon
        test_dataset['slide'] = self.slide
        test_dataset['lead'] = self.lead
        test_dataset['beta'] = self.beta
        return dataset, test_dataset


    def get_events_from_df(self,df_list):
        events = []
        for df in df_list:
            is_fail = self.rtf_dict[df.iloc[0][self.source_column]]
            if is_fail == 1:
                events.append(
                    [pd.to_datetime(df.iloc[-1][self.datetime_column]), "failure", df.iloc[0][self.source_column],
                     "failure"])
            else:
                events.append(
                    [pd.to_datetime(df.iloc[-1][self.datetime_column]), "reset", df.iloc[0][self.source_column],
                     "maintenance"])
        return events




def episodes_formulation(data,datetime_column,event_indicator=None,maintenance_column=None,failure_column=None,event_df=None,source_column='source',DIVIDER=3600):

    if event_df is None:
        if event_indicator is not None:
            #group by source and indicate the event:
            all_episodes = []
            all_run_to_failure = []
            all_sources = []
            original_s_has_f={}
            for source,group_df in data.groupby(source_column):
                group_df=group_df.sort_values(by=datetime_column).reset_index(drop=True)
                if len(group_df[event_indicator].unique())>2:
                    raise ValueError(f"event_indicator column must be binary (0 and 1) for each source. Source {source} has values {group_df[event_indicator].unique()}")
                if "RUL" not in group_df.columns:
                    maxdate = group_df[datetime_column].max()
                    group_df["RUL"] = [(maxdate - dtime).total_seconds() / DIVIDER for dtime in
                                       group_df[datetime_column]]

                all_run_to_failure.append(group_df.iloc[0][event_indicator])
                original_s_has_f[source]=group_df.iloc[0][event_indicator]==1  or original_s_has_f.get(source,False)
                all_episodes.append(group_df.drop(columns=[event_indicator]))
                all_sources.append(f"{source}_ep0")
            return all_episodes, all_run_to_failure,all_sources,original_s_has_f
        # check if maintenance_column and failure_column are in data
        if maintenance_column not in data.columns or failure_column not in data.columns:
            print("Warning: maintenance_column and failure_column or event column is not in data, we consider each source as run_to_failure.")
            all_episodes = []
            all_run_to_failure = []
            original_s_has_f = {}
            all_sources = []
            for source, group_df in data.groupby(source_column):
                group_df = group_df.sort_values(by=datetime_column).reset_index(drop=True)
                if "RUL" not in group_df.columns:
                    maxdate=group_df[datetime_column].max()
                    group_df["RUL"]=[(maxdate - dtime).total_seconds()/DIVIDER for dtime in group_df[datetime_column]]
                all_run_to_failure.append(1)
                original_s_has_f[source] = True or original_s_has_f.get(source, False)
                all_episodes.append(group_df)
                all_sources.append(f"{source}_ep0")
            return all_episodes, all_run_to_failure, all_sources, original_s_has_f
        else:
            event_data = data[[datetime_column,source_column,maintenance_column,failure_column]].copy()
            event_data=event_data.dropna(subset=[maintenance_column,failure_column],how='all')
            event_data=event_data[(event_data[maintenance_column]==1) | (event_data[failure_column]==1)]
    else:
        # check if maintenance_column and failure_column are in data
        if maintenance_column not in event_df.columns or failure_column not in event_df.columns:
            raise ValueError("maintenance_column and failure_column must be present in event_df.")
        # check if datetime_column, source_column are in event_df
        if datetime_column not in event_df.columns or source_column not in event_df.columns:
            raise ValueError("datetime_column and source_column must be present in event_df.")
        event_data = event_df[[datetime_column,source_column,maintenance_column,failure_column]].copy()

    # check if datetime_column, source_column are in data
    if datetime_column not in data.columns or source_column not in data.columns:
        raise ValueError("datetime_column and source_column must be present in data.")

    all_sources=[]
    all_episodes = []
    all_run_to_failure = []
    original_s_has_f = {}
    for source in data[source_column].unique():
        df_source=data[data[source_column]==source].copy()

        episodes, rtfs,new_sources = data_split_by_event(df_source,event_data[event_data[source_column]==source],
                                                         datetime_column,failure_column,maintenance_column,source_column,DIVIDER)
        all_episodes.extend(episodes)
        all_run_to_failure.extend(rtfs)
        original_s_has_f[source] = max(rtfs)==1 or original_s_has_f.get(source, False)
        all_sources.extend(new_sources)

    return all_episodes, all_run_to_failure,all_sources,original_s_has_f




def data_split_by_event(df_source,event_source,datetime_column,failure_column,maintenance_column,source_column='source',DIVIDER=3600):
    df_source.sort_values(by=datetime_column, inplace=True)
    df_source.reset_index(drop=True, inplace=True)
    event_source.sort_values(by=datetime_column, inplace=True)
    event_source.reset_index(drop=True, inplace=True)
    episodes = []
    rtfs = []
    new_sources=[]
    counter=0
    for idx, event_row in event_source.iterrows():
        event_time = event_row[datetime_column]
        found = False
        if event_row[failure_column] == 1:
            # Failure event
            found=True
            rtfs.append(1)
        elif event_row[maintenance_column] == 1:
            rtfs.append(0)
            found = True
        if found:
            if idx == 0:
                start_time = df_source[datetime_column].min()
            else:
                prev_event_time = event_source.loc[idx - 1, datetime_column]
                start_time = prev_event_time
            end_time = event_time
            episode = df_source[(df_source[datetime_column] > start_time) & (df_source[datetime_column] <= end_time)].copy()
            episode["RUL"]= [(episode[datetime_column].max() - dtime).total_seconds()/DIVIDER for dtime in episode[datetime_column]]
            episode[source_column]=f"{df_source.iloc[0][source_column]}_ep{counter}"
            episodes.append(episode)
            new_sources.append(f"{df_source.iloc[0][source_column]}_ep{counter}")
            counter += 1
    return episodes, rtfs, new_sources













