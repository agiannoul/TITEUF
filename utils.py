import random

import numpy as np
import pandas as pd

from OnlineADEngine.experiment.batch.RUL_experiment import SupervisedRULPdMExperiment
from OnlineADEngine.utils.dataset import Dataset


def run_train_val_test(dataset,test_dataset,method_class,param_space_dict_per_method,method_name,
                           preprocessor=None,pre_run=None,thresholder=None,
                           additional_params={},debug=False,datasetname="",optimization_param="IBS",maximize=False,MAX_RUNS=20):
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
                                            MAX_RUNS=MAX_RUNS, MAX_JOBS=1, INITIAL_RANDOM=1,optimization_param=optimization_param,
                          debug=debug,maximize=maximize,thresholder=thresholder,additional_parameters=additional_params)
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

    params=run_experiment(test_dataset, methods, [test_params], method_names,
                   experiments, experiment_names, preprocessor=preprocessor, mlflow_port=None,
                   thresholder=thresholder, additional_parameters=additional_params,
                   MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1, optimization_param=optimization_param, debug=True, maximize=maximize)
    return params['best_objective'], best_parames['best_params']


def encode_categoricals(df, cat_cols, mode="label"):
    """
    Encode categorical columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    cat_cols : list of str
        Column names to encode.
    mode : str, default="label"
        - "label": replaces each category with an integer (1, 2, 3, ...)
        - "onehot": expands categorical columns into one-hot encoded columns

    Returns
    -------
    pd.DataFrame
        DataFrame with encoded categorical columns.
    """
    df = df.copy()

    if mode == "onehot":
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    else:
        raise ValueError("mode must be either 'label' or 'onehot'")

    return df


def categorical_to_numerical(series, categories):
    category_series = {}

    # Include the NaN category explicitly
    all_categories = categories

    for cat, val in zip(all_categories, [ki for ki in range(1, len(categories) + 1)]):
        res = series.map({cat: val / len(categories)}).fillna(0)
        category_series[cat] = res
    return category_series


def create_new_binary_signal(timestamps, ori_signal, max_time_d):
    """
    Merge nearby '1' segments in a binary signal if their separation
    is less than or equal to `max_time_d`.

    Parameters
    ----------
    timestamps : pd.Series or array-like of datetime64
        The timestamps corresponding to the signal samples.
    ori_signal : pd.Series or array-like of 0/1 (and possibly NaN)
        The original binary signal.
    max_time_d : pd.Timedelta
        The maximum allowed time gap between consecutive '1' segments
        to be merged together.

    Returns
    -------
    pd.Series
        New binary signal with merged '1' segments.
    """
    # Convert to pandas Series
    ts = timestamps
    sig = ori_signal.fillna(0).astype(int)

    # Identify where the signal transitions (0→1 or 1→0)
    changes = sig.diff().fillna(0)

    # Find start and end indices of each '1' segment
    starts = ts[changes == 1].reset_index(drop=True)
    ends = ts[changes == -1].reset_index(drop=True)

    # Handle edge case if signal starts or ends with 1
    if sig.iloc[0] == 1:
        starts = pd.concat([pd.Series(ts.iloc[0]), starts], ignore_index=True)
    if sig.iloc[-1] == 1:
        ends = pd.concat([ends, pd.Series(ts.iloc[-1])], ignore_index=True)

    # Merge segments if time between consecutive ones ≤ max_time_d
    merged_starts, merged_ends = [starts.iloc[0]], []
    for i in range(1, len(starts)):
        gap = starts.iloc[i] - ends.iloc[i - 1]
        if gap <= max_time_d:
            # merge segments (extend last one)
            continue
        else:
            merged_ends.append(ends.iloc[i - 1])
            merged_starts.append(starts.iloc[i])
    merged_ends.append(ends.iloc[-1])

    # Build new signal initialized to zeros
    new_sig = pd.Series(0, index=ts.index, dtype=int)

    # Fill 1's for merged intervals
    for s, e in zip(merged_starts, merged_ends):
        mask = (ts >= s) & (ts <= e)
        new_sig.loc[mask] = 1

    return new_sig


def consecutive_occurencies(consecutive_maintenances):
    diffed = consecutive_maintenances.diff()
    starts = diffed[diffed < 0].count()
    if consecutive_maintenances.iloc[0] > 0:
        starts += 1
    print(starts)


def extract_context_segments(signal, df, n_context=10000):
    """
    Given a binary signal, extract the zero segments before and after
    each contiguous block of 1's. The number of samples taken before
    and after is limited to `n_context`, or fewer if the signal ends earlier.

    Parameters
    ----------
    signal : pd.Series or array-like
        Binary signal (0s and 1s).
    n_context : int, optional
        Maximum number of samples to include before and after each 1-run.

    Returns
    -------
    list[tuple]
        A list of tuples (start_idx, end_idx, segment) for each extracted
        zero-segment around a 1-run.
    """
    s = pd.Series(signal).fillna(0).astype(int)
    diff = s.diff().fillna(0)

    # Find start and end indices of each 1-block
    starts = s.index[diff == 1].tolist()
    ends = s.index[diff == -1].tolist()
    # Handle cases where signal starts/ends with 1
    if len(starts) > 0 and (len(ends) == 0 or starts[0] > ends[0]):
        ends = ends[1:]
        # ends.append(-1)
    if len(starts) > len(ends):
        starts = starts[:len(ends)]

    context_segments = []
    prev_end = 0
    pointer_i = -1
    for start, end in zip(starts, ends):
        pointer_i += 1
        # Before region
        # print(start)
        # print(end)
        # print("=======")
        start_before = max(prev_end, start - n_context)
        end_before = start
        if end_before > start_before:
            seg_before = df.iloc[start_before:end_before]
        else:
            continue
        prev_end = end
        # After region
        start_after = end + 1
        if len(starts) > pointer_i + 1:
            next_start = starts[pointer_i + 1]
        else:
            next_start = len(s)
        end_after = min(next_start, end + 1 + n_context)
        if end_after > start_after:
            seg_after = df.iloc[start_after:end_after]
        else:
            continue

        context_segments.append(([start_before, end_before], [start_after, end_after], seg_before, seg_after))

    return context_segments


def check_Nan(df):
    nan_ratio = df.isna().mean().sort_values(ascending=False)
    return nan_ratio



def load_SCANIA_with_failures(start_time, filepath="./Data/SCANIA/full_train_dataset_rtf.parquet", split_ratio=0.6,
                              validation_ratio=0.2, event=1):
    dfrtf = pd.read_parquet(filepath)
    dfrtf = dfrtf.sort_values(by=["vehicle_id", "time_step"])
    dfrtf = dfrtf.fillna(method='ffill')
    dfrtf.dropna(inplace=True)
    dfrtf['RUL'] = dfrtf['RUL'].round(1)
    categorical_cols = ['Spec_0', 'Spec_1', 'Spec_2', 'Spec_3', 'Spec_4', 'Spec_5', 'Spec_6', 'Spec_7']
    dfrtf['dates'] = start_time + pd.to_timedelta(dfrtf['time_step'], unit='h')
    dfrtf['event'] = event
    dfrtf.drop(columns=['time_step'], inplace=True)
    ## no stratified
    dfrtf = encode_categoricals(dfrtf, categorical_cols, mode="onehot")
    # print(dfrtf.shape)
    unvid = [vid for vid in dfrtf["vehicle_id"].unique()]
    # print(f"Total unique vehicle ids: {len(unvid)}")

    import random
    random.seed(42)
    random.shuffle(unvid)

    trainlimit = int(len(unvid) * split_ratio)
    val_limit = int(len(unvid) * (split_ratio + validation_ratio))
    for_train = unvid[:trainlimit]
    for_val = unvid[trainlimit:val_limit]
    for_test = unvid[val_limit:]

    dftrain = dfrtf[dfrtf["vehicle_id"].isin(for_train)]
    dftest = dfrtf[dfrtf["vehicle_id"].isin(for_test)]
    dfval = dfrtf[dfrtf["vehicle_id"].isin(for_val)]

    return dftrain, dfval, dftest


def load_dataset_SCANIA():
    start_time = pd.Timestamp("2025-01-01 00:00:00")

    fail_dftrain, fail_val, fail_dftest = load_SCANIA_with_failures(start_time, event=1)
    ce_dftrain, ce_dfval, ce_dftest = load_SCANIA_with_failures(start_time,
                                                                filepath="Data/SCANIA/full_train_dataset_ce.parquet",
                                                                event=0)

    # print("Columns in fail_dftrain but not in ce_dftrain:", set(fail_dftrain.columns) - set(ce_dftrain.columns))
    # print("Columns in ce_dftrain but not in fail_dftrain:", set(ce_dftrain.columns) - set(fail_dftrain.columns))
    for col in set(ce_dftrain.columns) - set(fail_dftrain.columns):
        fail_dftrain[col] = 0
        fail_val[col] = 0
        fail_dftest[col] = 0

    # print(fail_dftrain.shape)
    # print(fail_val.shape)
    # print(fail_dftest.shape)
    # print(ce_dftrain.shape)
    # print(ce_dfval.shape)
    # print(ce_dftest.shape)

    return fail_dftrain, fail_val, fail_dftest, ce_dftrain, ce_dfval, ce_dftest


def load_SACNIA_surv(keep_identifiers=False,keep_censored=True):
    fail_dftrain, fail_val, fail_dftest, ce_dftrain, ce_dfval, ce_dftest = load_dataset_SCANIA()
    return load_dataset_surv(fail_dftrain, fail_val, fail_dftest, ce_dftrain, ce_dfval, ce_dftest,
                             keep_identifiers=keep_identifiers,keep_censored=keep_censored)


def load_HNEI_SA(keep_identifiers=False,keep_censored=True):
    df = pd.read_csv("Data/HNEI_combined.csv")
    from sklearn.preprocessing import MinMaxScaler
    columns = [col for col in df.columns if col not in ["source", "Artificial_timestamp", "RUL", "event"]]
    scaler=MinMaxScaler()
    fitdata=df[df["source"].isin(["a","b", "c",  "d","e","f","g","n" ])]
    scaler.fit(fitdata[columns])
    df[columns]=scaler.transform(df[columns])
    handler = Dataset(data=df, datetime_column="Artificial_timestamp"
                      , train_sources=["a","b", "c",  "d","e","f","g","n" ],
                      val_sources=["l","j","o"],#"1_5","2_4","2_5"
                      test_sources=["t","s","p"],DIVIDER=3600*24*7) # "1_6","1_7","2_6","2_7"
    if keep_identifiers:
        source_to_keep="vehicle_id"
    else:
        source_to_keep=None
    dataset, test_dataset = handler.get_SA_dataset(keep_sources=source_to_keep)
    return dataset, test_dataset

def load_HNEI_rul(keep_identifiers=False,keep_censored=True):
    df = pd.read_csv("Data/HNEI_combined.csv")
    from sklearn.preprocessing import MinMaxScaler
    columns = [col for col in df.columns if col not in ["source", "Artificial_timestamp", "RUL", "event"]]
    scaler=MinMaxScaler()
    fitdata=df[df["source"].isin(["a","b", "c",  "d","e","f","g","n" ])]
    scaler.fit(fitdata[columns])
    df[columns]=scaler.transform(df[columns])
    handler = Dataset(data=df, datetime_column="Artificial_timestamp"
                      , train_sources=["a","b", "c",  "d","e","f","g","n" ],
                      val_sources=["l","j","o"],#"1_5","2_4","2_5"
                      test_sources=["t","s","p"],DIVIDER=3600*24*7) # "1_6","1_7","2_6","2_7"
    if keep_identifiers:
        source_to_keep="vehicle_id"
    else:
        source_to_keep=None
    dataset, test_dataset = handler.get_rul_dataset(keep_sources=source_to_keep)
    return dataset, test_dataset


def make_source_censored_at(source,seed,data,rul_column):
    random.seed(seed)

    df_source = data[data["source"] == source].copy()
    rul_cut = random.randint(df_source[rul_column].min() + 5, df_source[rul_column].max() - 5)
    failure_indices = df_source.index[df_source["event"] == 1].tolist()
    if len(failure_indices) == 0:
        return df_source
    df_source=df_source[df_source[rul_column]< rul_cut].copy()
    df_source["event"] = [0 for i in range(df_source.shape[0])]
    return df_source

def load_HNEI_censored(keep_identifiers=False,censore_sources=2,seed=1,rul_SA=None):
    df = pd.read_csv("Data/HNF/HNEI_combined.csv")
    train_sources=["a","b", "c", "d","e","f","g","n" ]
    from sklearn.preprocessing import MinMaxScaler
    columns = [col for col in df.columns if col not in ["source", "Artificial_timestamp", "RUL", "event"]]
    scaler = MinMaxScaler()
    fitdata = df[df["source"].isin(train_sources)]
    scaler.fit(fitdata[columns])
    df[columns] = scaler.transform(df[columns])

    cols=train_sources
    new_cols=[]
    for i in range(len(cols)):
        new_cols.append(cols[(i+seed)%len(cols)])
    for col in new_cols[:censore_sources]:
        df_source = make_source_censored_at(col, seed,data=df, rul_column="RUL")
        df = df[df["source"] != col]
        df = pd.concat([df,df_source],axis=0)
    handler = Dataset(data=df, datetime_column="Artificial_timestamp"
                      , train_sources=train_sources,
                      val_sources=["l", "j", "o"],  # "1_5","2_4","2_5"
                      test_sources=["t", "s", "p"], DIVIDER=3600 * 24 * 7)

    if keep_identifiers:
        source_to_keep = "vehicle_id"
    else:
        source_to_keep = None
    if rul_SA=="rul":
        dataset, test_dataset = handler.get_rul_dataset(keep_sources=source_to_keep)
    elif rul_SA=="sa":
        dataset, test_dataset = handler.get_SA_dataset(keep_sources=source_to_keep)
    else:
        raise ValueError("rul_SA must be either 'rul' or 'sa'")
    return dataset, test_dataset


def load_SCANIA_surv_no_censored(keep_identifiers=False):
    fail_dftrain, fail_val, fail_dftest, ce_dftrain, ce_dfval, ce_dftest = load_dataset_SCANIA()
    return load_dataset_surv(fail_dftrain, fail_val, fail_dftest, ce_dftrain, ce_dfval, ce_dftest,
                             keep_identifiers=keep_identifiers, keep_censored=False)

def load_dataset_surv(fail_dftrain, fail_val, fail_dftest, ce_dftrain, ce_dfval, ce_dftest,
                      keep_identifiers=False, id_col="vehicle_id", dates_column='dates',keep_censored=True):
    sampleN = len(fail_dftrain[id_col].unique())
    keep_censored_vehicles = ce_dftrain[id_col].sample(n=sampleN, random_state=42).unique()
    ce_dftrain = ce_dftrain[ce_dftrain[id_col].isin(keep_censored_vehicles)]

    # ce_dfval = ce_dfval.sample(n=500, random_state=42)
    if keep_censored:
        combined_dftrain = pd.concat([fail_dftrain, ce_dftrain], axis=0)
    else:
        keep_censored_vehicles = ce_dftrain[id_col].sample(n=1, random_state=42).unique()
        ce_dftrain = ce_dftrain[ce_dftrain[id_col].isin(keep_censored_vehicles)].sample(n=2, random_state=42)
        combined_dftrain = pd.concat([fail_dftrain, ce_dftrain], axis=0)
    # combined_dfval = pd.concat([fail_val, ce_dfval], axis=0).sample(frac=1, random_state=42)
    combined_dfval = fail_val

    sources_val = []
    target_data_list = []
    target_label_list = []
    for vehicle_id, group_df in combined_dfval.groupby(id_col):
        sources_val.append(str(vehicle_id))
        if not keep_identifiers:
            Xval, yval = df_to_x_y_surv(group_df, exclude_cols=["event", "RUL", id_col])
        else:
            Xval, yval = df_to_x_y_surv(group_df, exclude_cols=["event", "RUL"])
        target_data_list.append(Xval)
        target_label_list.append(yval)
    dataset = {}
    matches = {}
    for vid in combined_dfval[id_col].unique():
        matches[str(vid)] = "a"
    dataset['match_sources'] = matches
    dataset['target_sources'] = sources_val
    dataset['target_data'] = target_data_list
    dataset['target_labels'] = target_label_list

    if keep_identifiers == False:
        X, y = df_to_x_y_surv(combined_dftrain, exclude_cols=["event", "RUL", id_col])
        dataset['historic_data'] = [X]
    else:
        X, y = df_to_x_y_surv(combined_dftrain, exclude_cols=["event", "RUL"])
        dataset['historic_data'] = [X]
    dataset['historic_sources'] = ["a"]
    dataset['anomaly_labels'] = [y]
    dataset["dates"] = dates_column

    from OnlineADEngine.pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }
    dataset["event_preferences"] = event_preferences
    dataset["event_data"] = event_data
    dataset['predictive_horizon'] = None
    dataset['slide'] = None
    dataset['lead'] = None
    dataset['beta'] = None

    ##################### TEST DATASET ##############

    # cdftrain = pd.concat([combined_dftrain,combined_dfval])
    cdftrain = combined_dftrain
    # combined_dftest= pd.concat([fail_dftest, ce_dftest], axis=0).sample(frac=1, random_state=42)
    combined_dftest = fail_dftest

    # dftrain[num_cols] = scaler.transform(dftrain[num_cols])

    test_sources_val = []
    test_target_data_list = []
    test_target_label_list = []
    for vehicle_id, group_df in combined_dftest.groupby(id_col):
        test_sources_val.append(str(vehicle_id))
        if not keep_identifiers:
            Xval, yval = df_to_x_y_surv(group_df, exclude_cols=["event", "RUL", id_col])
        else:
            Xval, yval = df_to_x_y_surv(group_df, exclude_cols=["event", "RUL"])
        test_target_data_list.append(Xval)
        test_target_label_list.append(yval)

    test_dataset = {}
    test_matches = {}
    for vid in combined_dftest[id_col].unique():
        test_matches[str(vid)] = "a"

    test_dataset['match_sources'] = test_matches
    test_dataset['target_sources'] = test_sources_val
    test_dataset['target_data'] = test_target_data_list
    test_dataset['target_labels'] = test_target_label_list

    if keep_identifiers == False:
        X, y = df_to_x_y_surv(cdftrain, exclude_cols=["event", "RUL", id_col])
        test_dataset['historic_data'] = [X]
    else:
        X, y = df_to_x_y_surv(cdftrain, exclude_cols=["event", "RUL"])
        test_dataset['historic_data'] = [X]

    test_dataset['historic_sources'] = ["a"]
    test_dataset['anomaly_labels'] = [y]
    test_dataset["dates"] = dates_column

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }
    test_dataset["event_preferences"] = event_preferences
    test_dataset["event_data"] = event_data
    test_dataset['predictive_horizon'] = None
    test_dataset['slide'] = None
    test_dataset['lead'] = None
    test_dataset['beta'] = None

    return dataset, test_dataset


def df_to_x_y_surv(df: pd.DataFrame, exclude_cols=["event", "RUL", "vehicle_id", "dates"]):
    y = [(rul, ev) for ev, rul in zip(df["event"], df["RUL"])]
    X = df[df.columns.difference(exclude_cols)]
    return X, y


def combine_azure_data_():
    df_tel = pd.read_csv("Azure/PdM_telemetry.csv")
    df_tel["datetime"] = pd.to_datetime(df_tel["datetime"])
    df_tel["machineID"] = df_tel["machineID"].astype(str)
    models_ages = pd.read_csv("Azure/PdM_machines.csv")
    dict_map_age = {}
    for m_id, age, model in zip(models_ages["machineID"], models_ages["age"], models_ages["model"]):
        dict_map_age[str(m_id)] = (age, model)

    df_tel["age"] = [dict_map_age[m_id][0] for m_id in df_tel["machineID"]]
    df_tel["model"] = [dict_map_age[m_id][1] for m_id in df_tel["machineID"]]

    # "datetime","machineID","errorID"
    df_error = pd.read_csv("Azure/PdM_errors.csv")

    df_error["datetime"] = pd.to_datetime(df_error["datetime"])

    # df_tel["error"]=[0 for i in range(df_tel.shape[0])]
    df_error["machineID"] = df_error["machineID"].astype(str)
    df_tel["machineID"] = df_tel["machineID"].astype(str)
    # one-hot encode
    errors_oh = pd.get_dummies(df_error["errorID"], prefix="err")

    # append columns
    df_error = pd.concat([df_error[["datetime", "machineID"]], errors_oh], axis=1)
    df_error = (
        df_error
        .groupby(["machineID", "datetime"], as_index=False)
        .max()  # multiple errors at same time → OR (max) aggregation
    )
    df_tel['datetime'] = pd.to_datetime(df_tel['datetime'], utc=False)
    df_error['datetime'] = pd.to_datetime(df_error['datetime'], utc=False)

    df_tel = df_tel.sort_values(["datetime", "machineID"], kind="mergesort").reset_index(drop=True)
    df_error = df_error.sort_values(["datetime", "machineID"], kind="mergesort").reset_index(drop=True)

    df = pd.merge_asof(
        df_tel,
        df_error,
        on="datetime",
        by="machineID",
        direction="nearest",
        tolerance=pd.Timedelta("5min")  # optional; define the max allowed distance
    )
    df = df.fillna(0)
    df = df.sort_values(["machineID", "datetime"], kind="mergesort").reset_index(drop=True)

    df["event"] = [-1 for i in range(df.shape[0])]
    df["RUL"] = [-1 for i in range(df.shape[0])]

    maintenance = pd.read_csv("Azure/PdM_maint.csv")

    failure = pd.read_csv("Azure/PdM_failures.csv")

    maint = maintenance.copy()
    maint["event_type"] = "maintenance"
    maint["datetime"] = pd.to_datetime(maint["datetime"])

    fail = failure.copy()
    fail["event_type"] = "failure"
    fail["datetime"] = pd.to_datetime(fail["datetime"])

    common_rows = pd.merge(maint, fail, on=["machineID", "datetime"], how="inner")

    # Filter out common rows from maint
    maint = maint[
        ~maint.set_index(['machineID', 'datetime']).index.isin(common_rows.set_index(['machineID', 'datetime']).index)]

    events = pd.concat([maint, fail]).sort_values(["machineID", "datetime"])
    events["machineID"] = events["machineID"].astype(str)

    df = df.sort_values(["machineID", "datetime"])
    events = events.sort_values(["machineID", "datetime"])
    per_id = {}
    for m_id in events["machineID"].unique():
        m_ev = events[events["machineID"] == m_id]
        per_id[m_id] = [(1, date, f"{m_id}_{idd}") if ev == "failure" else (0, date, f"{m_id}_{idd}") for ev, date, idd
                        in zip(m_ev["event_type"], m_ev["datetime"], range(m_ev["event_type"].shape[0]))]
        per_id[m_id].append((0, df[df["machineID"] == m_id]["datetime"].max(), m_ev["event_type"].shape[0]))
        per_id[m_id].sort(key=lambda x: x[1])
    ruls = []
    evs = []
    vehicle_ids = []
    for m_id, date in zip(df["machineID"], df["datetime"]):
        done = False
        for tup in per_id[m_id]:
            if date <= tup[1]:
                ruls.append((tup[1] - date).total_seconds() / 3600)
                evs.append(tup[0])
                vehicle_ids.append(tup[2])
                done = True
                break
        if not done:
            print("PROBLEM")

    df["event"] = evs

    # RUL in hours
    df["RUL"] = ruls
    df["vehicle_id"] = vehicle_ids
    df = df.drop(["machineID"], axis=1)
    # print([col for col in df.columns])
    for col in ['err_error1', 'err_error2', 'err_error3', 'err_error4', 'err_error5']:
        df[col] = df.groupby('vehicle_id')[col].cumsum()
    df.to_csv("azure.csv")


def azure_split(df, event=1):
    dfrtf = df[df["event"] == event]
    dfrtf = dfrtf.sort_values(by=["vehicle_id", "datetime"])
    dfrtf = dfrtf.fillna(method='ffill')
    dfrtf.dropna(inplace=True)
    dfrtf['RUL'] = dfrtf['RUL'].round(1)
    ## no stratified
    # print(dfrtf.shape)
    unvid = [vid for vid in dfrtf["vehicle_id"].unique()]
    # print(f"Total unique vehicle ids: {len(unvid)}")

    import random
    random.seed(42)
    random.shuffle(unvid)
    split_ratio = 0.6
    validation_ratio = 0.2
    trainlimit = int(len(unvid) * split_ratio)
    val_limit = int(len(unvid) * (split_ratio + validation_ratio))
    for_train = unvid[:trainlimit]
    for_val = unvid[trainlimit:val_limit]
    for_test = unvid[val_limit:]

    dftrain = dfrtf[dfrtf["vehicle_id"].isin(for_train)]
    dftest = dfrtf[dfrtf["vehicle_id"].isin(for_test)]
    dfval = dfrtf[dfrtf["vehicle_id"].isin(for_val)]

    return dftrain, dfval, dftest


def read_azure(keep_identifiers=False,keep_censored=True):
    # ,datetime,machineID,volt,rotate,pressure,vibration,age,model,err_error1,err_error2,err_error3,err_error4,err_error5,event,RUL

    df = pd.read_csv("Data/azure.csv", index_col=0)
    df = pd.get_dummies(df, columns=["model"], prefix="model", drop_first=False)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["time_elapsed_from_maint"] = 1
    df["time_elapsed_from_maint"] = df.groupby('vehicle_id')["time_elapsed_from_maint"].cumsum()

    fail_dftrain, fail_val, fail_dftest = azure_split(df, event=1)
    ce_dftrain, ce_dfval, ce_dftest = azure_split(df, event=0)

    return load_dataset_surv(fail_dftrain, fail_val, fail_dftest, ce_dftrain, ce_dfval, ce_dftest,
                             keep_identifiers=keep_identifiers, id_col="vehicle_id", dates_column='datetime',keep_censored=keep_censored)


def read_azure_no_censored(keep_identifiers=False):
    # ,datetime,machineID,volt,rotate,pressure,vibration,age,model,err_error1,err_error2,err_error3,err_error4,err_error5,event,RUL

    df = pd.read_csv("Data/azure.csv", index_col=0)
    df = pd.get_dummies(df, columns=["model"], prefix="model", drop_first=False)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["time_elapsed_from_maint"] = 1
    df["time_elapsed_from_maint"] = df.groupby('vehicle_id')["time_elapsed_from_maint"].cumsum()

    fail_dftrain, fail_val, fail_dftest = azure_split(df, event=1)
    ce_dftrain, ce_dfval, ce_dftest = azure_split(df, event=0)

    return load_dataset_surv(fail_dftrain, fail_val, fail_dftest, ce_dftrain, ce_dfval, ce_dftest,
                             keep_identifiers=keep_identifiers, id_col="vehicle_id", dates_column='datetime',
                             keep_censored=False)

def read_azure_rul(keep_identifiers=False,use_scales=True):
    # ,datetime,machineID,volt,rotate,pressure,vibration,age,model,err_error1,err_error2,err_error3,err_error4,err_error5,event,RUL

    df = pd.read_csv("Data/azure.csv", index_col=0)
    df = pd.get_dummies(df, columns=["model"], prefix="model", drop_first=False)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["time_elapsed_from_maint"] = 1
    df["time_elapsed_from_maint"] = df.groupby('vehicle_id')["time_elapsed_from_maint"].cumsum()

    split_ratio = 0.6
    validation_ratio = 0.2

    dfrtf = df[df["event"] == 1]
    unvid = [vid for vid in dfrtf["vehicle_id"].unique()]
    # print(f"Total unique vehicle ids: {len(unvid)}")

    import random
    random.seed(42)
    random.shuffle(unvid)

    trainlimit = int(len(unvid) * split_ratio)
    val_limit = int(len(unvid) * (split_ratio + validation_ratio))
    for_train = unvid[:trainlimit]
    for_test = unvid[trainlimit:val_limit]
    dfrtf_or = dfrtf.copy()
    dftrain = dfrtf_or[dfrtf_or["vehicle_id"].isin(for_train)]

    cols_to_exclude = ['vehicle_id', 'RUL', 'event', 'datetime']
    num_cols = dftrain.columns.difference(cols_to_exclude)

    if use_scales:
        from sklearn.preprocessing import MinMaxScaler
        minmaxscaler = MinMaxScaler()
        dftrain[num_cols] = minmaxscaler.fit_transform(dftrain[num_cols])
        dfrtf[num_cols] = minmaxscaler.transform(dfrtf[num_cols])

    cols_to_drop = ["vehicle_id", "RUL"]
    dataset = {}
    matches = {}
    for vid in for_test:
        matches[str(vid)] = "a"
    dataset['match_sources'] = matches
    dataset['target_sources'] = [str(vid) for vid in for_test]
    if keep_identifiers == False:
        dataset['target_data'] = [
            dfrtf[dfrtf["vehicle_id"] == vid].drop(columns=cols_to_drop).reset_index(drop=True).copy()
            for vid in for_test]
    else:
        dataset['target_data'] = [
            dfrtf[dfrtf["vehicle_id"] == vid].drop(columns=["RUL"]).reset_index(drop=True).copy() for vid in
            for_test]
    dataset['target_labels'] = [dfrtf[dfrtf["vehicle_id"] == vid]["RUL"].values for vid in for_test]

    if keep_identifiers == False:
        dataset['historic_data'] = [dftrain.drop(columns=cols_to_drop)]
    else:
        dataset['historic_data'] = [dftrain.drop(columns=["RUL"])]
    dataset['historic_sources'] = ["a"]
    dataset['anomaly_labels'] = [dftrain["RUL"].values]
    dataset["dates"] = 'datetime'

    from OnlineADEngine.pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }
    dataset["event_preferences"] = event_preferences
    dataset["event_data"] = event_data
    dataset['predictive_horizon'] = None
    dataset['slide'] = None
    dataset['lead'] = None
    dataset['beta'] = None

    ##################### TEST DATASET ##############

    for_train = unvid[:val_limit]
    for_test = unvid[val_limit:]

    dftrain = dfrtf_or[dfrtf_or["vehicle_id"].isin(for_train)]
    dfrtf = dfrtf_or.copy()
    if use_scales:
        from sklearn.preprocessing import MinMaxScaler
        minmaxscaler = MinMaxScaler()
        dftrain[num_cols] = minmaxscaler.fit_transform(dftrain[num_cols])
        dfrtf[num_cols] = minmaxscaler.transform(dfrtf[num_cols])

    test_dataset = {}
    test_matches = {}
    for vid in for_test:
        test_matches[str(vid)] = "a"
    test_dataset['match_sources'] = test_matches
    test_dataset['target_sources'] = [str(vid) for vid in for_test]
    if keep_identifiers == False:
        test_dataset['target_data'] = [
            dfrtf[dfrtf["vehicle_id"] == vid].drop(columns=cols_to_drop).reset_index(drop=True)
            for vid in for_test]
    else:
        test_dataset['target_data'] = [
            dfrtf[dfrtf["vehicle_id"] == vid].drop(columns=["RUL"]).reset_index(drop=True) for vid in
            for_test]

    test_dataset['target_labels'] = [dfrtf[dfrtf["vehicle_id"] == vid]["RUL"].values for vid in for_test]

    if keep_identifiers == False:
        test_dataset['historic_data'] = [dftrain.drop(columns=cols_to_drop)]
    else:
        test_dataset['historic_data'] = [dftrain.drop(columns=["RUL"])]
    test_dataset['historic_sources'] = ["a"]
    test_dataset['anomaly_labels'] = [dftrain["RUL"].values]
    test_dataset["dates"] = 'datetime'

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }
    test_dataset["event_preferences"] = event_preferences
    test_dataset["event_data"] = event_data
    test_dataset['predictive_horizon'] = None
    test_dataset['slide'] = None
    test_dataset['lead'] = None
    test_dataset['beta'] = None

    return dataset, test_dataset



def nan_inf_summary(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    nan_cols = numeric_df.isna().sum()
    inf_cols = np.isinf(numeric_df).sum()

    summary = pd.DataFrame({
        "NaN_count": nan_cols,
        "Inf_count": inf_cols
    })
    summary = summary[(summary["NaN_count"] > 0) | (summary["Inf_count"] > 0)]
    return summary



def load_test(Max_limit=200,split_ratio=0.6,validation_ratio=0.2,keep_identifiers=False):
    dfrtf = pd.read_parquet("Data/SCANIA/full_train_dataset_rtf.parquet")
    dfrtf = dfrtf.sort_values(by=["vehicle_id", "time_step"])
    dfrtf = dfrtf.fillna(method='ffill')
    dfrtf.dropna(inplace=True)


    # if Max_limit is not None:
    #     dfrtf['RUL'] = dfrtf['RUL'].clip(upper=Max_limit)
    # print(f"unique RUL: {len(dfrtf['RUL'].unique())}")
    dfrtf['RUL'] = dfrtf['RUL'].round(1)
    # print(f"unique RUL: {len(dfrtf['RUL'].unique())}")
    categorical_cols = ['Spec_0', 'Spec_1', 'Spec_2', 'Spec_3', 'Spec_4', 'Spec_5', 'Spec_6', 'Spec_7']
    start_time = pd.Timestamp("2025-01-01 00:00:00")
    dfrtf['dates'] = start_time + pd.to_timedelta(dfrtf['time_step'], unit='h')

    ## no stratified
    dfrtf = encode_categoricals(dfrtf, categorical_cols, mode="onehot")
    # print(dfrtf.shape)
    unvid = [vid for vid in dfrtf["vehicle_id"].unique()]
    # print(f"Total unique vehicle ids: {len(unvid)}")

    import random
    random.seed(42)
    random.shuffle(unvid)

    trainlimit=int(len(unvid) * split_ratio)
    val_limit=int(len(unvid) * (split_ratio + validation_ratio))
    for_train = unvid[:trainlimit]
    for_test = unvid[trainlimit:val_limit]

    for_estimation= unvid[val_limit:]
    # print(for_estimation)

    dftrain = dfrtf[dfrtf["vehicle_id"].isin(for_train)]

    # #vehicle_id      time_step RUL ce
    from sklearn.preprocessing import StandardScaler

    cols_to_exclude = ['vehicle_id', 'time_step', 'RUL', 'ce','dates']
    num_cols = dftrain.columns.difference(cols_to_exclude)

    # scaler = StandardScaler()
    # dftrain[num_cols] = scaler.fit_transform(dftrain[num_cols])


    dataset={}
    matches={}
    for vid in for_test:
        matches[str(vid)]="a"
    dataset['match_sources']= matches
    dataset['target_sources']=[str(vid) for vid in for_test]
    if keep_identifiers == False:
        dataset['target_data'] = [dfrtf[dfrtf["vehicle_id"]==vid].drop(columns=["vehicle_id","time_step","RUL"]).reset_index(drop=True) for vid in for_test]
    else:
        dataset['target_data'] = [dfrtf[dfrtf["vehicle_id"]==vid].drop(columns=["time_step","RUL"]).reset_index(drop=True) for vid in for_test]
    dataset['target_labels']=[dfrtf[dfrtf["vehicle_id"]==vid]["RUL"].values for vid in for_test]

    if keep_identifiers == False:
        dataset['historic_data']= [dftrain.drop(columns=["vehicle_id", "time_step", "RUL"])]
    else:
        dataset['historic_data']= [dftrain.drop(columns=["time_step", "RUL"])]
    dataset['historic_sources']=["a"]
    dataset['anomaly_labels']=[dftrain["RUL"].values]
    dataset["dates"]='dates'

    from OnlineADEngine.pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }
    dataset["event_preferences"] = event_preferences
    dataset["event_data"] = event_data
    dataset['predictive_horizon'] = None
    dataset['slide'] = None
    dataset['lead'] = None
    dataset['beta'] = None

    ##################### TEST DATASET ##############

    for_train = unvid[:val_limit]
    for_test = unvid[val_limit:]
    dftrain = dfrtf[dfrtf["vehicle_id"].isin(for_train)]
    # dftrain[num_cols] = scaler.transform(dftrain[num_cols])

    test_dataset={}
    test_matches={}
    for vid in for_test:
        test_matches[str(vid)]="a"
    test_dataset['match_sources']= test_matches
    test_dataset['target_sources']=[str(vid) for vid in for_test]
    if keep_identifiers == False:
        test_dataset['target_data'] = [dfrtf[dfrtf["vehicle_id"]==vid].drop(columns=["vehicle_id","time_step","RUL"]).reset_index(drop=True) for vid in for_test]
    else:
        test_dataset['target_data'] = [dfrtf[dfrtf["vehicle_id"]==vid].drop(columns=["time_step","RUL"]).reset_index(drop=True) for vid in for_test]

    test_dataset['target_labels']=[dfrtf[dfrtf["vehicle_id"]==vid]["RUL"].values for vid in for_test]

    if keep_identifiers == False:
        test_dataset['historic_data']= [dftrain.drop(columns=["vehicle_id", "time_step", "RUL"])]
    else:
        test_dataset['historic_data']= [dftrain.drop(columns=["time_step", "RUL"])]
    test_dataset['historic_sources']=["a"]
    test_dataset['anomaly_labels']=[dftrain["RUL"].values]
    test_dataset["dates"]='dates'

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }
    test_dataset["event_preferences"] = event_preferences
    test_dataset["event_data"] = event_data
    test_dataset['predictive_horizon'] = None
    test_dataset['slide'] = None
    test_dataset['lead'] = None
    test_dataset['beta'] = None


    return dataset, test_dataset


from OnlineADEngine.RunExperiment import run_experiment


def run_rul_train_val_test(dataset,test_dataset,method_class,param_space_dict_per_method,method_name,
                           optimization_param="mape",preprocessor=None,pre_run=None ,dataset_name="",mlflow_port=5011,debug_test=True):
    experiments = [SupervisedRULPdMExperiment]
    experiment_names = [f'RUL {dataset_name}']


    methods = [method_class]

    method_names = [f"{method_name} Train-Val"]


    from OnlineADEngine.preprocessing.record_level.default import DefaultPreProcessor

    if preprocessor is None:
        preprocessor = DefaultPreProcessor

    if pre_run is not None:
        correct_pre_run = {}
        for key in pre_run:
            if key.startswith("method_"):
                correct_pre_run[key]=pre_run[key]
            else:
                correct_pre_run[f"method_{key}"]=pre_run[key]
        params=[{'best_params': correct_pre_run, 'best_objective': None}]
        print(f"PRE RUN MODE: {pre_run}")
    else:
        params=run_experiment(dataset, methods, param_space_dict_per_method, method_names,
                                                experiments, experiment_names,preprocessor=preprocessor,mlflow_port=mlflow_port,
                                                MAX_RUNS=20, MAX_JOBS=1, INITIAL_RANDOM=1,optimization_param=optimization_param,debug=False,maximize=False)

    best_parames= params[0]
    #
    test_params = {}
    for key in param_space_dict_per_method[0]:
        test_params[key]=[best_parames['best_params'][f'method_{key}']]
    test_params["save_model"]=[True]
    method_names = [f"{method_name}"]
    print(f"test params {test_params}")
    run_experiment(test_dataset, methods, [test_params], method_names,
                                                experiments, experiment_names,preprocessor=preprocessor,mlflow_port=mlflow_port,
                                                MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1,optimization_param=optimization_param,debug=debug_test,maximize=False)




if __name__ == "__main__":
    # combine_azure_data_()
    dataset, test_dataset = read_azure()

