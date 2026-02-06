import pickle
import subprocess
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import read_azure, load_SACNIA_surv, load_HNEI_censored

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PdM evaluation plots"
    )

    parser.add_argument(
        "--plot",
        type=str,
        choices=[
            "global",
            "calibration",
            "bins",
            "datasets",
            "hm_sigmoid",
            "family_box",
            "latex_table",
        ],
        required=True,
        help="Plot to generate"
    )

    return parser.parse_args()

def is_port_in_use(host, port):
    """Check if a given port is in use on the specified host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def run_mlflow_server(mlflow_port=5000):
    """Function to run MLflow server if it's not already running."""
    host = "127.0.0.1"
    port = mlflow_port

    if is_port_in_use(host, port):
        print(f"MLflow server is already running at http://{host}:{port}.")
    else:
        print("Starting MLflow server...")
        # subprocess.Popen(["export","MLFLOW_TRACKING_URI=sqlite:///mlruns.db"])
        subprocess.run([
            "mlflow", "server",
            "--host", "0.0.0.0",
            "--port", f"{mlflow_port}",
            "--backend-store-uri", "./mlrunsp"
        ])
        print(f"MLflow server started at http://{host}:{port}.")


def get_runtime(host="http://127.0.0.1:5000/", datasetname="SCANIA", SA_or_RUL="SA"):
    from mlflow.tracking import MlflowClient
    import math

    client = MlflowClient()

    # Get experiments ending in _TEST
    test_experiments = client.search_experiments(
        filter_string=f"name LIKE '%{SA_or_RUL}% {datasetname} %'"
    )
    test_experiments2 = client.search_experiments(
        filter_string=f"name LIKE '%{SA_or_RUL}% {datasetname.upper()} %'"
    )
    exps_ids=[exp.experiment_id for exp in test_experiments]
    test_experiments2=[exp for exp in test_experiments2 if exp.experiment_id not in exps_ids]

    if datasetname == "SCANIA":
        test_experiments3 = client.search_experiments(
            filter_string=f"name LIKE '%{SA_or_RUL}% SACNIA %'"
        )
        exps_ids2 = [exp.experiment_id for exp in test_experiments2]
        test_experiments2 = test_experiments2 + [exp for exp in test_experiments3 if exp.experiment_id not in exps_ids2]
    test_experiments=test_experiments+test_experiments2
    print(len(test_experiments))
    rows = []
    all_metric_keys = set()

    for exp in test_experiments:
        if "Train-Val" in exp.name or "CNNDeepHIT" in exp.name:
            continue
        if exp.name.replace("_TEST", "").replace(" TEST", "").replace(" TEST ", "").replace("My RUL experiment ", "").replace(
                "My SA experiment ", "").replace(SA_or_RUL, "").strip().upper().replace("CNIA","SCANIA") in [rr["experiment_name"].strip().upper() for rr in rows]:
            continue
        # Search runs
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
        )

        # ---------------------------------------------
        # Keep only runs where "IBR" metric exists and is NOT NaN
        # ---------------------------------------------
        valid_runs = []
        for r in runs:
            metrics = r.data.metrics

            if "inference_time" in metrics:
                valid_runs.append(r)
        # If no valid run, skip experiment
        if len(valid_runs) == 0:
            continue

        # ---------------------------------------------
        # Select the most recent run (by start time)
        # ---------------------------------------------
        valid_runs.sort(key=lambda x: x.info.start_time, reverse=True)
        best_run = valid_runs[0]

        metrics = best_run.data.metrics
        new_metrics = {}
        for key in metrics.keys():
            if SA_or_RUL == "RUL":
                if key == "IBR":
                    new_metrics["IBS_HM"] = metrics[key]
                elif key == "IBR_SigT10":
                    new_metrics["IBS"] = metrics[key]
                elif "IBR" in metrics.keys() and key == "Max_brier":
                    new_metrics["Max_brier_HM"] = metrics[key]
                elif key == "Max_brier_SigT10":
                    new_metrics["Max_brier"] = metrics[key]
                else:
                    new_metrics[key] = metrics[key]
            else:
                if key == "IBR":
                    new_metrics["IBS"] = metrics[key]
                else:
                    new_metrics[key] = metrics[key]
        all_metric_keys.update(new_metrics.keys())

        row = {
            "experiment_name": exp.name.replace("_TEST", "").replace("My RUL experiment ", "").replace(
                "My SA experiment ", "").replace(SA_or_RUL, "").strip(),
            "run_id": best_run.info.run_id,
            **new_metrics
        }
        rows.append(row)
    return rows

def get_run_ids(host="http://127.0.0.1:5000/", datasetname="SCANIA", SA_or_RUL="SA"):
    from mlflow.tracking import MlflowClient
    import math

    client = MlflowClient()

    # Get experiments ending in _TEST
    test_experiments = client.search_experiments(
        filter_string=f"name LIKE '%{SA_or_RUL}% {datasetname} %'"
    )
    test_experiments2 = client.search_experiments(
        filter_string=f"name LIKE '%{SA_or_RUL}% {datasetname.upper()} %'"
    )
    exps_ids=[exp.experiment_id for exp in test_experiments]
    test_experiments2=[exp for exp in test_experiments2 if exp.experiment_id not in exps_ids]

    if datasetname == "SCANIA":
        test_experiments3 = client.search_experiments(
            filter_string=f"name LIKE '%{SA_or_RUL}% SACNIA %'"
        )
        exps_ids2 = [exp.experiment_id for exp in test_experiments2]
        test_experiments2 = test_experiments2 + [exp for exp in test_experiments3 if exp.experiment_id not in exps_ids2]
    test_experiments=test_experiments+test_experiments2
    print(len(test_experiments))
    rows = []
    all_metric_keys = set()

    for exp in test_experiments:
        if "Train-Val" in exp.name or "CNNDeepHIT" in exp.name:
            continue
        if exp.name.replace("_TEST", "").replace(" TEST", "").replace(" TEST ", "").replace("My RUL experiment ", "").replace(
                "My SA experiment ", "").replace(SA_or_RUL, "").strip().upper().replace("CNIA","SCANIA") in [rr["experiment_name"].strip().upper() for rr in rows]:
            continue
        # Search runs
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
        )

        # ---------------------------------------------
        # Keep only runs where "IBR" metric exists and is NOT NaN
        # ---------------------------------------------
        valid_runs = []
        for r in runs:
            metrics = r.data.metrics

            if "IBR" in metrics and metrics["IBR"] is not None and not math.isnan(metrics["IBR"]):
                valid_runs.append(r)
            elif "IBS" in metrics and metrics["IBS"] is not None and not math.isnan(metrics["IBS"]):
                valid_runs.append(r)
        # If no valid run, skip experiment
        if len(valid_runs) == 0:
            continue

        # ---------------------------------------------
        # Select the most recent run (by start time)
        # ---------------------------------------------
        valid_runs.sort(key=lambda x: x.info.start_time, reverse=True)
        best_run = valid_runs[0]

        metrics = best_run.data.metrics
        new_metrics = {}
        for key in metrics.keys():
            if SA_or_RUL == "RUL":
                if key == "IBR":
                    new_metrics["IBS_HM"] = metrics[key]
                elif key == "IBR_SigT10":
                    new_metrics["IBS"] = metrics[key]
                elif "IBR" in metrics.keys() and key == "Max_brier":
                    new_metrics["Max_brier_HM"] = metrics[key]
                elif key == "Max_brier_SigT10":
                    new_metrics["Max_brier"] = metrics[key]
                else:
                    new_metrics[key] = metrics[key]
            else:
                if key == "IBR":
                    new_metrics["IBS"] = metrics[key]
                else:
                    new_metrics[key] = metrics[key]
        all_metric_keys.update(new_metrics.keys())

        row = {
            "experiment_name": exp.name.replace("_TEST", "").replace("My RUL experiment ", "").replace(
                "My SA experiment ", "").replace(SA_or_RUL, "").strip(),
            "run_id": best_run.info.run_id,
            **new_metrics
        }
        rows.append(row)
    return rows

def get_exps(datasetname="SCANIA",SA_or_RUL="SA"):
    from mlflow.tracking import MlflowClient
    import math

    client = MlflowClient()

    # Get experiments ending in _TEST
    test_experiments = client.search_experiments(
        filter_string=f"name LIKE '%{SA_or_RUL}% {datasetname} %'"
    )
    test_experiments2 = client.search_experiments(
        filter_string=f"name LIKE '%{SA_or_RUL}% {datasetname.upper()} %'"
    )
    exps_ids = [exp.experiment_id for exp in test_experiments]
    test_experiments2 = [exp for exp in test_experiments2 if exp.experiment_id not in exps_ids]

    if datasetname == "SCANIA":
        test_experiments3 = client.search_experiments(
            filter_string=f"name LIKE '%{SA_or_RUL}% SACNIA %'"
        )
        exps_ids2 = [exp.experiment_id for exp in test_experiments2]
        test_experiments2 = test_experiments2 + [exp for exp in test_experiments3 if exp.experiment_id not in exps_ids2]
    test_experiments = test_experiments + test_experiments2
    print(len(test_experiments))
    rows = []
    all_metric_keys = set()

    for exp in test_experiments:
        if "Train-Val" in exp.name or "CNNDeepHIT" in exp.name:
            continue
        if exp.name.replace("_TEST", "").replace(" TEST", "").replace(" TEST ", "").replace("My RUL experiment ",
                                                                                            "").replace(
                "My SA experiment ", "").replace(SA_or_RUL, "").strip().upper().replace("CNIA","SCANIA") in [rr["experiment_name"].strip().upper() for rr in
                                                                            rows]:
            continue
        # Search runs
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
        )

        # ---------------------------------------------
        # Keep only runs where "IBR" metric exists and is NOT NaN
        # ---------------------------------------------
        valid_runs = []
        for r in runs:
            metrics = r.data.metrics

            if "IBR" in metrics and metrics["IBR"] is not None and not math.isnan(metrics["IBR"]):
                valid_runs.append(r)
            elif "IBS" in metrics and metrics["IBS"] is not None and not math.isnan(metrics["IBS"]):
                valid_runs.append(r)

        # If no valid run, skip experiment
        if len(valid_runs) == 0:
            continue

        # ---------------------------------------------
        # Select the most recent run (by start time)
        # ---------------------------------------------
        valid_runs.sort(key=lambda x: x.info.start_time, reverse=True)
        best_run = valid_runs[0]

        metrics = best_run.data.metrics

        from datetime import datetime
        start_time = best_run.info.start_time  # ms since epoch
        end_time = best_run.info.end_time  # ms since epoch (can be None if still running
        duration_sec = (end_time - start_time) / 1000 if end_time else None
        metrics["runtime"]= duration_sec
        new_metrics={}
        for key in metrics.keys():
            if SA_or_RUL == "RUL":
                if key=="IBR":
                    new_metrics["IBS_HM"]=metrics[key]
                elif key=="IBR_SigT10":
                    new_metrics["IBS"]=metrics[key]
                elif "IBR" in metrics.keys() and key=="Max_brier":
                    new_metrics["Max_brier_HM"]=metrics[key]
                elif  key=="Max_brier_SigT10":
                    new_metrics["Max_brier"]=metrics[key]
                else:
                    new_metrics[key]=metrics[key]
            else:
                if key=="IBR":
                    new_metrics["IBS"]=metrics[key]
                else:
                    new_metrics[key]=metrics[key]
        all_metric_keys.update(new_metrics.keys())

        row = {
            "experiment_name": exp.name.replace("_TEST", "").replace("My RUL experiment ", "").replace(
                "My SA experiment ", "").replace(SA_or_RUL, "").strip(),
            "run_id": best_run.info.run_id,
            **new_metrics
        }
        rows.append(row)

    # Build dataframe
    df = pd.DataFrame(rows)

    # Ensure all metric columns exist
    for key in all_metric_keys:
        if key not in df.columns:
            df[key] = None

    # Reorder columns
    cols = ["experiment_name", "run_id"] + sorted(all_metric_keys)
    df = df[cols]

    # Save CSV
    df.to_csv(f"Results/mlflow_{datasetname}_{SA_or_RUL}.csv", index=False)

    print("Saved:", f"Results/mlflow_{datasetname}_{SA_or_RUL}.csv")
    print("Shape:", df.shape)

def plot_runtime_bar(
    df,
    metric_col,
    time_col,
    dataset_col,
    method_col,
    normalize=False,
    logy=False,
    figsize=(12, 6),
    title=None
):
    import seaborn as sns
    plt.figure(figsize=(15, 6))
    unique_ds = sorted(df[dataset_col].unique())
    base_colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, 3))  # only 3 colors
    palette = {ds: base_colors[i % 3] for i, ds in enumerate(unique_ds)}
    sns.barplot(
        data=df,
        x=method_col,
        y=time_col,
        hue=dataset_col,
        palette=palette,
        errorbar=None  # since each dataset-method pair has one value
    )
    plt.yscale('log')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Training Time (seconds)")
    plt.xlabel("Method")
    # plt.title("Model Fit Time per Method across Datasets")
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.show()

def runtimeplot_comparison():

    p = Path("Results/runtime_res.pkl")
    try:
        with p.open("rb") as f:
            res = pickle.load(f)
    except Exception:
        p.parent.mkdir(parents=True, exist_ok=True)
        res = get_runtime_res()
        with p.open("wb") as f:
            pickle.dump(res, f)

    res["name"]= [name.split("_")[0].replace("SKTIME","").replace("CATBOOST","CatBoost").replace("XGBOOST","XGBoost").
                  replace("DEEPHIT", "DeepHit").replace("COXPH", "CoxPH").replace("INCEPTIONTIME", "Inception\nTime").
                  replace("ELASTICNET", "ElasticNet").replace("GRADIENTBOOSTING", "Gradient\nBoosting").replace("RANDOMFOREST", "RF")
                  for name in res["name"]]
    df=pd.DataFrame(res)
    df["dataset"]=[name.split(" ")[0] for name in df["name"].values]
    df["method_name"]=[name.split(" ")[1] for name in df["name"].values]

    plot_runtime_bar(
        df,
        "IBS",
        "fit_time",
        "dataset",
        "method_name",
        normalize=False,
        logy=False,
        figsize=(12, 6),
        title=None
    )
    df["approach"]=["SA" if method_nema in ["CoxPH","DeepHit","RDSM","RSF","Gradient\nBoosting"] else "RUL"  for method_nema in df["method_name"].values]
    METRIC = "mdape"
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), sharey="row")

    dfazur= df[df["dataset"]=="AZURE"]
    dhnei= df[df["dataset"]=="HNEI"]
    dfScania= df[df["dataset"]=="SCANIA"]

    pareto_inner_2("mdape", ax[0][0], dfazur, legend_=True)
    pareto_inner_2("IBS", ax[1][0], dfazur, legend_=False)
    ax[0][0].set_ylabel("MdAPE", fontsize=14)
    ax[1][0].set_ylabel("IBS", fontsize=14)

    # ax[1][0].set_xlabel("Total runtime", fontsize=14)
    ax[1][1].set_xlabel("Inference time (s)", fontsize=14)
    # ax[1][2].set_xlabel("Total runtime", fontsize=14)

    pareto_inner_2("mdape", ax[0][1], dhnei, legend_=False)
    pareto_inner_2("IBS", ax[1][1], dhnei, legend_=False)

    pareto_inner_2("mdape", ax[0][2], dfScania, legend_=False)
    pareto_inner_2("IBS", ax[1][2], dfScania, legend_=False)

    ax[0][0].set_title("AZURE", fontsize=12)
    ax[0][1].set_title("HNEI", fontsize=12)
    ax[0][2].set_title("SCANIA", fontsize=12)

    plt.tight_layout()
    plt.show()




def get_runtime_res():
    rows = get_runtime(datasetname="SCANIA", SA_or_RUL="RULRT")
    rows2 = get_runtime(datasetname="Azure", SA_or_RUL="RULRT")
    rows3 = get_runtime(datasetname="HNEI", SA_or_RUL="RULRT")

    rowSA = get_runtime(datasetname="SCANIA05", SA_or_RUL="SA")
    rowSA2 = get_runtime(datasetname="AZURE05", SA_or_RUL="SA")
    rowSA3 = get_runtime(datasetname="HNEI05", SA_or_RUL="SA")

    dfRUL = pd.DataFrame(rows + rows2 + rows3)
    dfSA = pd.DataFrame(rowSA + rowSA2 + rowSA3)
    df = pd.concat([dfRUL, dfSA], ignore_index=True)

    try:
        dfHM = pd.read_csv("Results/mlflow_Azure_RUL.csv")
        dfHM2 = pd.read_csv("Results/mlflow_SCANIA_RUL.csv")
        dfHM3 = pd.read_csv("Results/mlflow_HNEI_RUL.csv")

        dfHM4 = pd.read_csv("Results/mlflow_SCANIA_SA.csv")
        dfHM5 = pd.read_csv("Results/mlflow_Azure_SA.csv")
        dfHM6 = pd.read_csv("Results/mlflow_HNEI_SA.csv")
    except:
        get_exps(datasetname="Azure", SA_or_RUL="RUL")
        get_exps(datasetname="SCANIA", SA_or_RUL="RUL")
        get_exps(datasetname="HNEI", SA_or_RUL="RUL")
        get_exps(datasetname="Azure", SA_or_RUL="SA")
        get_exps(datasetname="SCANIA", SA_or_RUL="SA")
        get_exps(datasetname="HNEI", SA_or_RUL="SA")
        dfHM = pd.read_csv("Results/mlflow_Azure_RUL.csv")
        dfHM2 = pd.read_csv("Results/mlflow_SCANIA_RUL.csv")
        dfHM3 = pd.read_csv("Results/mlflow_HNEI_RUL.csv")

        dfHM4 = pd.read_csv("Results/mlflow_SCANIA_SA.csv")
        dfHM5 = pd.read_csv("Results/mlflow_Azure_SA.csv")
        dfHM6 = pd.read_csv("Results/mlflow_HNEI_SA.csv")
    df2 = pd.concat([dfHM, dfHM2, dfHM3, dfHM6, dfHM5, dfHM4], ignore_index=True)
    df2["experiment_name"] = [name.upper().replace("CNIA", "SCANIA").replace(" TEST", "") for name in
                              df2["experiment_name"].values]
    df["experiment_name"] = [name.upper().replace("RUL", "").replace("05", "") for name in df["experiment_name"].values]
    res = {"name": [], "inference_time": [], "fit_time": [], "IBS": [], "mdape": []}
    for name in df["experiment_name"].values:
        res["name"].append(name)
        inf_time = df[df["experiment_name"] == name]["inference_time"].values[0]
        fit_time = df[df["experiment_name"] == name]["fit_time"].values[0]
        res["inference_time"].append(inf_time)
        res["fit_time"].append(fit_time)
        ibs = df2[df2["experiment_name"] == name]["IBS"].values[0]
        mdape = df2[df2["experiment_name"] == name]["mdape"].values[0]
        res["IBS"].append(ibs)
        res["mdape"].append(mdape)
    return res
def HM_vs_Sigmoid():
    import pandas as pd
    try:
        dfHM=pd.read_csv("Results/mlflow_Azure_RUL.csv")
        dfHM2=pd.read_csv("Results/mlflow_SCANIA_RUL.csv")
        dfHM3=pd.read_csv("Results/mlflow_HNEI_RUL.csv")
    except:
        get_exps(datasetname="Azure",SA_or_RUL="RUL")
        get_exps(datasetname="SCANIA", SA_or_RUL="RUL")
        get_exps(datasetname="HNEI", SA_or_RUL="RUL")
        dfHM=pd.read_csv("Results/mlflow_Azure_RUL.csv")
        dfHM2=pd.read_csv("Results/mlflow_SCANIA_RUL.csv")
        dfHM3=pd.read_csv("Results/mlflow_HNEI_RUL.csv")


    dfHM = pd.concat([dfHM, dfHM2,dfHM3], ignore_index=True)

    HM_IBRs=dfHM["IBS_HM"].values
    HM_MBS=dfHM["Max_brier_HM"].values


    RUL_IBRs = dfHM["IBS"].values
    RUL_MBS = dfHM["Max_brier"].values

    from scipy import stats
    names = [name.split(" ")[-1].replace("sktime", "").replace("RUL", "_").replace("RandomForest", "RF").split("_")[0] for name in dfHM["experiment_name"].values]
    # Wilcoxon signed-rank test (non-parametric alternative)
    datasets = [namee.split(" ")[0] for namee in  dfHM["experiment_name"].values]
    for dname,name,hm,sg,hmM,sgM in zip(datasets,names,HM_IBRs,RUL_IBRs,HM_MBS,RUL_MBS):
        print(f"{dname} &{name} & {hm:.3f} & {sg:.3f} & {hmM:.3f} & {sgM:.3f}\\\\")

    print("============")
    res = stats.wilcoxon(HM_IBRs,RUL_IBRs,  alternative='greater')
    # print("=== IBS test: Sigmoid better than Hard Mapping? ==== \n")
    # for name,hm,sg in zip( names,HM_IBRs,RUL_IBRs):
    #     print(f"{name}: HM IBS={hm:.4f} vs Sigmoid IBS={sg:.4f}")
    print(f"\nIBS test: Sigmoid better than Hard Mapping -> pvalue={res.pvalue:.4f}\n\n")

    # print("=== MBS test: Sigmoid better than Hard Mapping? ==== \n")
    # for name,hm,sg in zip( names,HM_MBS,RUL_MBS):
    #     print(f"{name}: HM MBS={hm:.4f} vs Sigmoid MBS={sg:.4f}")
    res = stats.wilcoxon(HM_MBS,RUL_MBS, alternative='greater')
    print(f"\nMBS test: Sigmoid better than Hard Mapping -> pvalue={res.pvalue:.4f}\n\n")
    from matplotlib.colors import TwoSlopeNorm
    x = np.arange(len(names))

    delta_mbs = [a - b for a, b in zip(HM_MBS, RUL_MBS)]
    delta_ibs = [a - b for a, b in zip(HM_IBRs, RUL_IBRs)]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # --- Helper for symmetric limits ---
    max_abs = max(
        max(abs(np.array(delta_mbs))),
        max(abs(np.array(delta_ibs)))
    )

    norm1 = TwoSlopeNorm(vmin=-max(abs(np.array(delta_mbs))), vcenter=0.0, vmax=max(abs(np.array(delta_mbs))))
    cmap = plt.cm.coolwarm

    # --- Plot 1: MBS ---
    colors_mbs = cmap(norm1(delta_mbs))
    ax[0].bar(x, delta_mbs, color=colors_mbs)
    ax[0].axhline(0)

    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=45, ha='right')
    ax[0].set_ylabel("MBS Difference (Hard Mapping − Sigmoid)")
    ax[0].set_title("MBS Δ")

    # --- Plot 2: IBS ---
    norm2 = TwoSlopeNorm(vmin=- max(abs(np.array(delta_ibs))), vcenter=0.0, vmax= max(abs(np.array(delta_ibs))))

    colors_ibs = cmap(norm2(delta_ibs))
    ax[1].bar(x, delta_ibs, color=colors_ibs)
    ax[1].axhline(0)

    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, rotation=45, ha='right')
    ax[1].set_title("IBS Δ")

    plt.tight_layout()
    plt.show()

def make_plots(ax,filename1="Results/mlflow_SCANIA_SA.csv",filename2="Results/mlflow_SCANIA_RUL.csv",sortind2=2,dataseName=""):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    dfSA=pd.read_csv(filename1)

    SA_names=dfSA["experiment_name"].values
    SA_IBRs=dfSA["IBS"].values
    SA_MBS=dfSA["Max_brier"].values
    highlightSA=[1 for _ in SA_names]

    dfRUL= pd.read_csv(filename2)

    RUL_names = dfRUL["experiment_name"].values
    RUL_IBRs = dfRUL["IBS"].values
    RUL_MBS = dfRUL["Max_brier"].values
    highlightRUL = [0 for _ in RUL_names]

    combined_names = list(SA_names) + list(RUL_names)
    combined_IBRs = list(SA_IBRs) + list(RUL_IBRs)
    combined_MBS = list(SA_MBS) + list(RUL_MBS)
    combined_HL = list(highlightSA) + list(highlightRUL)



    plot_dual_bars(ax[0],combined_names, combined_IBRs, combined_MBS,combined_HL,sortind=1)

    SA_names = dfSA["experiment_name"].values
    SA_MAPE = dfSA["mape"].values
    SA_MdAPE = dfSA["mdape"].values
    highlightSA = [1 for _ in SA_names]


    RUL_names = dfRUL["experiment_name"].values
    RUL_MAPE = dfRUL["mape"].values
    RUL_MdAPE = dfRUL["mdape"].values
    highlightRUL = [0 for _ in RUL_names]

    combined_names = list(SA_names) + list(RUL_names)
    combined_MAPE = list(SA_MAPE) + list(RUL_MAPE)
    combined_MdAPE = list(SA_MdAPE) + list(RUL_MdAPE)
    combined_HL = list(highlightSA) + list(highlightRUL)

    plot_dual_bars(ax[1],combined_names, combined_MdAPE,combined_MAPE ,combined_HL,label2="MAPE",label1="MdAPE",
                   color1='#d35400',color2='#f39c12',sortind=sortind2)

    ax[1].set_xlabel(dataseName)

def plot_dual_bars(ax,names, IBRs, MBSs,combined_HL,label2='MBS',label1='IBR',color1='#1f4e79',color2='#87ceeb',sortind=2):
        import matplotlib.pyplot as plt
        import numpy as np
        names=[name.split(" ")[-1].replace("sktime","").replace("RUL","_").replace("RandomForest","RF").split("_")[0] for name in names]
        # Sort data based on MBS values
        sorted_data = sorted(zip(names, IBRs, MBSs,combined_HL), key=lambda x: x[sortind], reverse=False)
        names, IBRs, MBSs,combined_HL = zip(*sorted_data)

        x = np.arange(len(names))  # the label locations
        width = 0.8  # the width of the bars

        labeltopot = "IBS" if label1 == "IBR" else label1
        # Plot bars
        bars2 = ax.bar(x, MBSs, width, label=label2, color=color2, edgecolor=color1)  # Darker blue
        bars1 = ax.bar(x, IBRs, width, label=labeltopot, color=color1, edgecolor=color1)  # Lighter blue
        if label1 == 'IBR':
            offset=0.03
        else:
            offset=0.2
        for bar,bar2 in zip(bars1,bars2):
            height = bar.get_height()
            height2 = bar2.get_height()
            if height < height2:
                height2 = max(height2, height + offset)
            else:
                height = max(height, height2 + offset)

            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)


            ax.text(bar2.get_x() + bar2.get_width() / 2, height2, f'{height2:.3f}', ha='center', va='bottom', fontsize=9)

        # for bar in bars2:

        # Add labels, title, and custom x-axis tick labels
        ax.set_xticks(x)
        ax.set_xticklabels(
            names,
            rotation=30,
            ha='right'
        )

        ticks = ax.get_xticklabels()

        # Ensure color_names matches the number of ticks
        for tick, flag in zip(ticks, combined_HL):
            tick.set_color("blue" if flag == 1 else "black")
        ax.legend()


def general_latex_table():
    try:
        dfHM = pd.read_csv("Results/mlflow_Azure_RUL.csv")
        dfHM2 = pd.read_csv("Results/mlflow_SCANIA_RUL.csv")
        dfHM3 = pd.read_csv("Results/mlflow_HNEI_RUL.csv")

        dfHM4 = pd.read_csv("Results/mlflow_SCANIA_SA.csv")
        dfHM5 = pd.read_csv("Results/mlflow_Azure_SA.csv")
        dfHM6 = pd.read_csv("Results/mlflow_HNEI_SA.csv")
    except:
        get_exps(datasetname="Azure", SA_or_RUL="RUL")
        get_exps(datasetname="SCANIA", SA_or_RUL="RUL")
        get_exps(datasetname="HNEI", SA_or_RUL="RUL")
        get_exps(datasetname="Azure", SA_or_RUL="SA")
        get_exps(datasetname="SCANIA", SA_or_RUL="SA")
        get_exps(datasetname="HNEI", SA_or_RUL="SA")
        dfHM = pd.read_csv("Results/mlflow_Azure_RUL.csv")
        dfHM2 = pd.read_csv("Results/mlflow_SCANIA_RUL.csv")
        dfHM3 = pd.read_csv("Results/mlflow_HNEI_RUL.csv")

        dfHM4 = pd.read_csv("Results/mlflow_SCANIA_SA.csv")
        dfHM5 = pd.read_csv("Results/mlflow_Azure_SA.csv")
        dfHM6 = pd.read_csv("Results/mlflow_HNEI_SA.csv")

    df = pd.concat([dfHM, dfHM2, dfHM3, dfHM6, dfHM5, dfHM4], ignore_index=True)
    df["c_index"]=[ cbest if np.isnan(cin) else cin for cin,cbest in zip( df["c_index"].values, df["c_index_best"].values) ]
    df["dataset"] = [name.split(" ")[0] for name in df["experiment_name"].values]
    df["method"] = [name.split(" ")[1].replace("sktime","") for name in df["experiment_name"].values]
    methods_to_show = ["ResNet", "FCN", "CNN", "InceptionTime", "LSTMFCN","XGBoost", "CatBoost_W_", "TABPFNv2", "RandomForest", "ElasticNet","RDSM", "RSF", "CoxPH", "GradientBoosting", "DeepHit"]
    metrics_to_show=["IBS", "Max_brier", "mape", "mdape","c_index"]
    metrics_to_display={"IBS":"IBS", "Max_brier":"MBS", "mape":"MAPE", "mdape":"MdAPE","c_index":"C-Index"}
    table_=df_to_latex_table_loops(
        df,
        methods_to_show,
        metrics_to_show,
        metrics_to_display,
        datasets_to_show=None,
        float_format="%.3f",
        caption=None,
        label=None
    )
    print(table_)

def df_to_latex_table_loops(
    df,
    methods_to_show,
    metrics_to_show,
    metrics_to_display,
    datasets_to_show=None,
    float_format="{:.3f}",
    caption=None,
    label=None,
):
    # Infer datasets if not provided
    if datasets_to_show is None:
        datasets_to_show = sorted(df["dataset"].unique())

    # Filter dataframe
    df_filt = df[df["method"].isin(methods_to_show) & df["dataset"].isin(datasets_to_show)]

    # Build LaTeX header
    num_metrics = len(metrics_to_show)
    header = "\\begin{table}[ht]\n\\centering\n"
    if caption:
        header += f"\\caption{{{caption}}}\n"
    if label:
        header += f"\\label{{{label}}}\n"

    col_format = "l" + "c" * (len(datasets_to_show) * num_metrics)
    header += f"\\begin{{tabular}}{{{col_format}}}\n\\toprule\n"

    # First header row: datasets
    header += "Method"
    for ds in datasets_to_show:
        header += f" & \\multicolumn{{{num_metrics}}}{{c}}{{{ds}}}"
    header += " \\\\\n"

    # Second header row: metrics
    header += " "
    for _ in datasets_to_show:
        for m in metrics_to_show:
            header += f" & {metrics_to_display[m]}"
    header += " \\\\\n\\midrule\n"

    # Body rows
    body = ""
    for method in methods_to_show:
        body += method
        for ds in datasets_to_show:
            df_md = df_filt[(df_filt["method"] == method) & (df_filt["dataset"] == ds)]

            for metric in metrics_to_show:
                if not df_md.empty and metric in df_md.columns:
                    val = df_md[metric].values[0]
                    try:
                        body += f" & {float(val):.3f}"
                    except (TypeError, ValueError):
                        body += f" & {val}"
                else:
                    body += " & --"
        body += " \\\\\n"

    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}"

    return header + body + footer



def per_Category():
    try:
        dfHM = pd.read_csv("Results/mlflow_Azure_RUL.csv")
        dfHM2 = pd.read_csv("Results/mlflow_SCANIA_RUL.csv")
        dfHM3 = pd.read_csv("Results/mlflow_HNEI_RUL.csv")

        dfHM4 = pd.read_csv("Results/mlflow_SCANIA_SA.csv")
        dfHM5 = pd.read_csv("Results/mlflow_Azure_SA.csv")
        dfHM6 = pd.read_csv("Results/mlflow_HNEI_SA.csv")
    except:
        get_exps(datasetname="Azure", SA_or_RUL="RUL")
        get_exps(datasetname="SCANIA", SA_or_RUL="RUL")
        get_exps(datasetname="HNEI", SA_or_RUL="RUL")
        get_exps(datasetname="Azure", SA_or_RUL="SA")
        get_exps(datasetname="SCANIA", SA_or_RUL="SA")
        get_exps(datasetname="HNEI", SA_or_RUL="SA")
        dfHM = pd.read_csv("Results/mlflow_Azure_RUL.csv")
        dfHM2 = pd.read_csv("Results/mlflow_SCANIA_RUL.csv")
        dfHM3 = pd.read_csv("Results/mlflow_HNEI_RUL.csv")

        dfHM4 = pd.read_csv("Results/mlflow_SCANIA_SA.csv")
        dfHM5 = pd.read_csv("Results/mlflow_Azure_SA.csv")
        dfHM6 = pd.read_csv("Results/mlflow_HNEI_SA.csv")


    df=pd.concat([dfHM, dfHM2, dfHM3, dfHM6,dfHM5,dfHM4], ignore_index=True)

    df["dataset"]=[name.split(" ")[0].replace("CNIA","SCANIA") for name in df["experiment_name"].values]
    df["method"]=[name.split(" ")[1] for name in df["experiment_name"].values]


    metric = "IBS"
    metric2 = "mdape"
    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(2, len(datasets), figsize=(9, 3), sharey=False)
    innter_box(axes[0], datasets, df, metric=metric)
    innter_box(axes[1], datasets, df, metric=metric2, second=True)
    plt.tight_layout()
    plt.show()


def innter_box(axes,datasets,df,metric="IBS",second=False):
    SA = ["RDSM", "RSF", "CoxPH", "GradientBoosting", "DeepHit"]
    tab = ["XGBoost", "CatBoost_W_", "TABPFNv2", "RandomForest", "ElasticNet"]
    tsm = ["sktimeResNet", "sktimeFCN", "sktimeCNN", "sktimeInceptionTime", "sktimeLSTMFCN"]
    for ax, dataset in zip(axes, datasets):
        df_dataset = df[df["dataset"] == dataset]

        sktime_vals = df_dataset[df_dataset["method"].isin(tsm)][metric].dropna().values
        tab_vals = df_dataset[df_dataset["method"].isin(tab)][metric].dropna().values
        SA_vals = df_dataset[df_dataset["method"].isin(SA)][metric].dropna().values

        cmap = plt.cm.coolwarm
        colors = cmap(np.linspace(0.2, 0.8, 3))
        if second:
            box = ax.boxplot(
                [sktime_vals, tab_vals, SA_vals],
                labels=["RUL\nTime series.", "RUL\nTabular.", "SA"],
                showfliers=True,
                patch_artist=True,
                widths = 0.6,
            )
        else:
            box = ax.boxplot(
                [sktime_vals, tab_vals, SA_vals],
                showfliers=True,
                patch_artist=True,
                widths=0.6,
                labels=['', '', ''],
            )
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(2)

        for median in box['medians']:
            median.set_color('black')

        for whisker in box['whiskers']:
            whisker.set_color('black')
        for cap in box['caps']:
            cap.set_color('black')

        if not second:
            ax.set_title(f"{dataset}")
        # ax.set_xlabel("Method Family")
        # ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    if second:
        axes[0].set_ylabel("MdAPE", fontdict={'size': 12})
    else:
        axes[0].set_ylabel(metric,fontdict={'size':12})





def figure_global():
    fig, ax = plt.subplots(nrows=2,ncols=3, figsize=(10, 6))

    try:
        dfHM=pd.read_csv("Results/mlflow_Azure_RUL.csv")
        dfHM2=pd.read_csv("Results/mlflow_SCANIA_RUL.csv")
        dfHM3=pd.read_csv("Results/mlflow_HNEI_RUL.csv")

        dfHM2=pd.read_csv("Results/mlflow_SCANIA_SA.csv")
        dfHM2=pd.read_csv("Results/mlflow_Azure_SA.csv")
        dfHM2=pd.read_csv("Results/mlflow_HNEI_SA.csv")
    except:
        get_exps(datasetname="Azure",SA_or_RUL="RUL")
        get_exps(datasetname="SCANIA", SA_or_RUL="RUL")
        get_exps(datasetname="HNEI", SA_or_RUL="RUL")
        get_exps(datasetname="Azure", SA_or_RUL="SA")
        get_exps(datasetname="SCANIA", SA_or_RUL="SA")
        get_exps(datasetname="HNEI", SA_or_RUL="SA")
    make_plots(ax=ax[:,0],filename1="Results/mlflow_SCANIA_SA.csv", filename2="Results/mlflow_SCANIA_RUL.csv", sortind2=1,dataseName="(a) SCANIA")
    make_plots(ax=ax[:,1],filename1="Results/mlflow_Azure_SA.csv", filename2="Results/mlflow_Azure_RUL.csv", sortind2=1,dataseName="(b) AZURE")
    make_plots(ax=ax[:,2],filename1="Results/mlflow_HNEI_SA.csv", filename2="Results/mlflow_HNEI_RUL.csv", sortind2=1,dataseName="(c) HNEI")

    # Show plot
    plt.tight_layout()
    plt.show()

import mlflow
import json
import os
import tempfile

def load_mdape_from_run(client,run_id,metric="MdAPE"):
    artifact_path=f"survival_{metric.lower()}_bins.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = client.download_artifacts(
            run_id=run_id,
            path=artifact_path,
            dst_path=tmpdir
        )

        with open(local_path, "r") as f:
            data = json.load(f)

    # column index lookup (robust)
    mdape_idx = data["columns"].index(metric)

    # extract MdAPE values in order
    return [row[mdape_idx] for row in data["data"]]

def mdape_diffs(ax,datasetname="Azure",url="127.0.0.1",mlflow_port=5000):

    client = mlflow.tracking.MlflowClient()
    artifact_path="MdAPE"
    rows=get_run_ids(host=url,SA_or_RUL="RUL",datasetname=datasetname)
    RUL_methods=[]
    rul_names=[]
    for row in rows:
        run_id=row["run_id"]
        mdape_bins=load_mdape_from_run(client,run_id,metric=artifact_path)
        RUL_methods.append(mdape_bins)
        rul_names.append(row["experiment_name"])

    rows2=get_run_ids(host=url,SA_or_RUL="SA",datasetname=datasetname)
    SA_methods = []
    SA_names = []
    for row in rows2:
        run_id=row["run_id"]
        mdape_bins=load_mdape_from_run(client,run_id,metric=artifact_path)
        SA_methods.append(mdape_bins)
        SA_names.append(row["experiment_name"])

    print("ok")

    DTSM=[mdape for mdape,name in zip(RUL_methods,rul_names) if is_DTSM(name)]
    DTSM_names=[name for mdape,name in zip(RUL_methods,rul_names) if is_DTSM(name)]
    analysis(ax[0],DTSM,DTSM_names,color='Reds')

    classical=[mdape for mdape,name in zip(RUL_methods,rul_names) if not is_DTSM(name) and "Rocket" not in name]
    classical_names=[name for mdape,name in zip(RUL_methods,rul_names) if not is_DTSM(name) and "Rocket" not in name]

    analysis(ax[1],classical,classical_names,color='Blues')
    # analysis(RUL_methods,rul_names)

    analysis(ax[2],SA_methods,SA_names,color="Greens")
    ax[0].set_ylabel(f"{artifact_path} {datasetname}",fontsize=14)
    ax[0].set_ylabel(f"{artifact_path} {datasetname}", fontsize=14)
    ax[0].tick_params(axis='y', labelsize=14)
    ax[0].tick_params(axis='x', labelsize=14)
    ax[1].tick_params(axis='x', labelsize=14)
    ax[2].tick_params(axis='x', labelsize=14)
    ax[1].set_xlabel(f"RUL regimes bins", fontsize=14)

def is_DTSM(name):
    if "CNN" in name or "FCN" in name or "ResNet" in name or "InceptionTime" in name :#or "Rocket" in name:
        return True
    return False


def analysis(ax, lists_of_bins, names, color=None):
    namesN = [name.split(" ")[-1].replace("sktime", "").replace("RUL", "_").replace("RandomForest", "RF").split("_")[0]
              for name in names]

    # Sort based on the first element of each bin
    sorted_data = sorted(zip(lists_of_bins, namesN), key=lambda x: x[0][0], reverse=True)
    lists_of_bins, namesN = zip(*sorted_data)

    for i, (bin, name) in enumerate(zip(lists_of_bins, namesN)):
        meanb = np.mean(bin)
        stdb = np.std(bin)
        minb = min(bin)
        normalized_bins = [b for b in bin]
        normalized_bins.append(bin[-1])
        # Use variants of the same color
        if color is not None:
            variant_color = plt.cm.get_cmap(color)(0.25 + 0.75 * i / len(lists_of_bins))
            ax.step(range(len(normalized_bins)), normalized_bins, where='post', label=name, color=variant_color)
            ax.fill_between(range(len(normalized_bins)), normalized_bins, step='post', color=variant_color)
            x = [nr for nr in range(len(normalized_bins))]
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(tick * 10)}%" for tick in range(len(x))], rotation=90, ha='center')
            # ax.set_xticklabels([f"{int(tick * 10)}%" for tick in x], rotation=90, ha='center')
        else:
            ax.step(range(len(normalized_bins)), normalized_bins, where='mid', label=name)

    ax.legend(fontsize=12)


def calibrated_threshold_analysis():

    rows=get_run_ids(SA_or_RUL="SA",datasetname="SCANIA05")
    rows.extend(get_run_ids(SA_or_RUL="SA",datasetname="Azure05"))
    rows.extend(get_run_ids(SA_or_RUL="SA",datasetname="HNEI05"))
    df05 = pd.DataFrame(rows)
    print([nam for nam in df05["experiment_name"].values])
    df=pd.read_csv("Results/mlflow_SCANIA_SA.csv")
    df2=pd.read_csv("Results/mlflow_Azure_SA.csv")
    df3=pd.read_csv("Results/mlflow_HNEI_SA.csv")
    df=pd.concat([df,df2,df3],ignore_index=True)


    wins=0
    losse=0
    increases=[]
    # metric="mape"
    metric="mape"
    df["experiment_name"]=[nam.replace(" TEST","",).replace("SCANIA","CNIA").replace("CNIA","SCANIA").replace("Azure","AZURE") for nam in df["experiment_name"].values]
    df05["experiment_name"]=[nam.replace("Azure","AZURE") for nam in df05["experiment_name"].values]
    all05=[]
    allc=[]
    for name_t in df05["experiment_name"].values:
        name=name_t.replace("TEST ","").replace("05","").strip()
        if name not in df["experiment_name"].values:
            print(f"Missing:{name}")
            print([nam for nam in df["experiment_name"].values])
            continue
        m05=df05[df05["experiment_name"]==name_t][metric].values[0]
        m=df[df["experiment_name"]==name][metric].values[0]
        print(m)
        allc.append(m)
        all05.append(m05)
        print(m05)
        print(f"=={name_t}==")
        if m<m05:
            wins+=1
        else:
            losse+=1
        increases.append((m-m05)/m05*100)
    from scipy import stats
    res=stats.wilcoxon(all05, allc, alternative='greater')
    print(f"\nCalibrated is better than fixed 0.5 -> pvalue={res.pvalue:.4f}\n\n")
    print("Wins:",wins,"out of",df.shape[0], "loses:", losse)
    print("Average increase:",sum(increases)/len(increases))
    print("Median increase:",np.median(increases))
    # for mape05,mape in zip(df05[metric],df[metric]):
    #     if mape<mape05:
    #         wins+=1
    #     increases.append((mape-mape05)/mape05*100)
    # print("Wins:",wins,"out of",df.shape[0])
    # print("Average increase:",sum(increases)/len(increases))

def plot_for_initial_bins(mlflow_port):
    fig,ax = plt.subplots(nrows=3,ncols=3, figsize=(4, 6),sharey=True)

    mdape_diffs(ax[0], datasetname="Azure", url="127.0.0.1",mlflow_port=mlflow_port)
    mdape_diffs(ax[1], datasetname="SCANIA", url="127.0.0.1",mlflow_port=mlflow_port)
    mdape_diffs(ax[2], datasetname="HNEI", url="127.0.0.1",mlflow_port=mlflow_port)
    
    ax[0][0].set_title("(a) Time-series\nregression",fontsize=16)
    ax[0][2].set_title("(c) Survival\nAnalysis",fontsize=16)
    ax[0][1].set_title("(b) Tabular\nregression",fontsize=16)
    plt.show()

def get_labels_of_all_sources(labs):
    all_labels = []
    for source_labels in labs:
        all_labels.extend(source_labels)
    return all_labels

def datasets_plots():

    plt.subplot(133)
    dataset, test_dataset = read_azure()
    inner_datasets_plots(dataset, test_dataset,first=True,name="Azure")
    plt.subplot(131)
    dataset, test_dataset = load_SACNIA_surv()
    inner_datasets_plots(dataset, test_dataset,name="SCANIA")
    plt.subplot(132)
    dataset, test_dataset = load_HNEI_censored(keep_identifiers=False,censore_sources=0,seed=0,rul_SA="sa")
    inner_datasets_plots(dataset, test_dataset, name="HNEI")
    plt.show()

def datasets_plots_cens():
    combinations=[(2,0),(2,2),(2,4),(2,6),(2,3),(4,0),(4,2),(4,4),(4,6),(4,3),(6,0),(6,2),(6,4),(6,6),(6,3)]
    fig,ax=plt.subplots(nrows=5,ncols=3,figsize=(15,9))
    c=-1
    for j in range(3):
        for i in range(5):
            c+=1
            comb= combinations[c]
            dataset, test_dataset =  dataset, test_dataset = load_HNEI_censored(keep_identifiers=False,censore_sources=comb[0],seed=comb[1],rul_SA="sa")
            inner_datasets_plots_ax(ax[i][j],dataset, test_dataset,j==0,name=f"HNEI censored {100*comb[0]/8.0:.0f}%",show_x=i==4)

    plt.show()


def inner_datasets_plots_ax(ax,dataset, test_dataset,first=False,name="Azure",show_x=False):



    train_labels =get_labels_of_all_sources(dataset['target_labels'])
    val_labels=get_labels_of_all_sources(dataset['anomaly_labels'])
    test_labels = get_labels_of_all_sources(test_dataset['anomaly_labels'])


    data=test_labels + train_labels +val_labels

    values_0 = [v[0] for v in data if v[1] == 0]
    values_1 = [v[0] for v in data if v[1] == 1]
    colors= [plt.cm.coolwarm(0.2), plt.cm.coolwarm(0.8)]
    ax.hist([values_0, values_1], bins=30, stacked=True, label=["Censored", "Failure"], color=colors)
    if len(values_0)+len(values_1)>1000*30:
        ax.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x / 1000)}k'))
    else:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.set_title(f"{name}", fontsize=12)
    if show_x:
        ax.set_xlabel("Time-to-event (or RUL)", fontsize=12)
    if first:
        ax.set_ylabel("Frequency", fontsize=12)
    ax.legend()
    print("ok")


def inner_datasets_plots(dataset, test_dataset,first=False,name="Azure"):



    train_labels =get_labels_of_all_sources(dataset['target_labels'])
    val_labels=get_labels_of_all_sources(dataset['anomaly_labels'])
    test_labels = get_labels_of_all_sources(test_dataset['anomaly_labels'])


    data=test_labels + train_labels +val_labels

    values_0 = [v[0] for v in data if v[1] == 0]
    values_1 = [v[0] for v in data if v[1] == 1]
    colors= [plt.cm.coolwarm(0.2), plt.cm.coolwarm(0.8)]
    plt.hist([values_0, values_1], bins=30, stacked=True, label=["Censored", "Failure"], color=colors)
    if len(values_0)+len(values_1)>1000*30:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x / 1000)}k'))
    else:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.title(f"{name}", fontsize=12)
    plt.xlabel("Time-to-event (or RUL)", fontsize=12)
    if first:
        plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    print("ok")

def pareto_inner_2(METRIC,ax,dff,legend_=False):
    pareto_df=dff.copy()
    pareto_df['metric_value'] = pareto_df[METRIC]
    pareto_df['duration'] = pareto_df["inference_time"]

    pareto_df['algorithm'] = [name.split(" ")[-1].replace("sktime", "").replace("RUL", "_").replace("RandomForest", "RF").split("_")[0] for name
     in pareto_df["method_name"].values]
    approaches = pareto_df["approach"].unique()
    selected_metric_display = METRIC.replace("_", " ").title()
    approach_colors = {
        'SA': plt.cm.coolwarm(0.2),
        'RUL': plt.cm.coolwarm(0.8)
    }

    edge_color="black"
    bg_color="#ffffff"
    unique_sybmols=['o','s','D','^','v','P','X','*','h','8','<','>','p','H','^']
    match_sumbols={}
    for i,alg in enumerate(sorted(pareto_df['algorithm'].unique())):
        match_sumbols[alg]=unique_sybmols[i%len(unique_sybmols)]

    from paretoset import paretoset

    if len(pareto_df) > 1:
        pareto_data = pareto_df[['duration', 'metric_value']].values
        pareto_data_transformed = np.column_stack([
            pareto_data[:, 0],  # duration (minimize)
            pareto_data[:, 1]  # negative metric (maximize)
        ])

        mask = paretoset(pareto_data_transformed, sense=["min", "min"])
        pareto_df['is_pareto'] = mask
    else:
        pareto_df['is_pareto'] = True

    # ax.set_facecolor('#1B2C3D')

    # Plot non-Pareto points
    non_pareto = pareto_df[~pareto_df['is_pareto']]
    if len(non_pareto) > 0:
        for approach in approaches:
            approach_data = non_pareto[non_pareto['approach'] == approach]
            for algorithm in approach_data['algorithm'].unique():
                algo_data = approach_data[approach_data['algorithm'] == algorithm]
                if not algo_data.empty:
                    ax.scatter(algo_data['duration'], algo_data['metric_value'],
                               c=approach_colors.get(approach, '#275CE6'), s=150,
                               alpha=0.6, #linewidths=2, #alpha=0.6,
                               # label=f'{approach} (Non-Pareto)',
                               label=f'{algorithm}',
                               marker=match_sumbols[algorithm],
                               zorder=2)

    # Plot Pareto optimal points
    pareto_optimal = pareto_df[pareto_df['is_pareto']].sort_values('duration')
    if len(pareto_optimal) > 0:
        for approach in approaches:
            approach_data = pareto_optimal[pareto_optimal['approach'] == approach]
            for algorithm in approach_data['algorithm'].unique():
                algo_data = approach_data[approach_data['algorithm'] == algorithm]
                if not algo_data.empty:
                    ax.scatter(algo_data['duration'], algo_data['metric_value'],
                               c=approach_colors.get(approach, '#275CE6'), s=200,
                               alpha=0.6,
                               label=f'{algorithm}', zorder=3, marker=match_sumbols[algorithm])

        # Draw Pareto front line
        if len(pareto_optimal) > 1:
            ax.plot(pareto_optimal['duration'], pareto_optimal['metric_value'],
                    color=edge_color, linewidth=2, linestyle='--',
                    alpha=0.7, zorder=1, label='Pareto Front')

    # Annotate all points
    changepoint=1
    for idx, row in pareto_df.iterrows():
        changepoint=changepoint*-1
        label = f"{row['algorithm']}"
        # ax.annotate(label,
        #             (row['duration'], row['metric_value']),
        #             xytext=(8, 8), textcoords='offset points',
        #             fontsize=8, color=edge_color, fontweight='bold',
        #             bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color,
        #                       edgecolor=edge_color, alpha=0.8))

    # ax.set_xlabel('Duration (seconds)', fontsize=12, fontweight='bold', color=edge_color)
    # ax.set_ylabel(f'{selected_metric_display}',
    #               fontsize=12, fontweight='bold', color=edge_color)


    # Set x-axis to logarithmic scale
    ax.set_xscale('log')

    ax.tick_params(axis='x', colors=edge_color, labelsize=10)
    ax.tick_params(axis='y', colors=edge_color, labelsize=10)
    if legend_:
        # Add legend
        legend = ax.legend(loc='lower left',bbox_to_anchor=(0, 1.2), fontsize=12, framealpha=0.9, ncol=6)
        legend.get_frame().set_facecolor(bg_color)
        legend.get_frame().set_edgecolor(edge_color)
        for text in legend.get_texts():
            text.set_color(edge_color)

    ax.grid(alpha=0.2, linestyle='--', color=edge_color)
    ax.set_axisbelow(True)

def pareto_inner(METRIC,ax,filename1="Results/mlflow_SCANIA_SA.csv", filename2="Results/mlflow_SCANIA_RUL.csv",legend_=False):
    dfRUL=pd.read_csv(filename2)
    dfRUL=dfRUL[dfRUL["experiment_name"].str.contains("Rocket")==False]
    dfSA=pd.read_csv(filename1)

    dfRUL["approach"]=["RUL" for _ in range(dfRUL.shape[0])]
    dfSA["approach"]=["SA" for _ in range(dfSA.shape[0])]
    dff=pd.concat([dfRUL,dfSA],ignore_index=True)
    pareto_df=dff.copy()
    pareto_df['metric_value'] = pareto_df[METRIC]
    pareto_df['duration'] = pareto_df["runtime"]

    pareto_df['algorithm'] = [name.split(" ")[-1].replace("sktime", "").replace("RUL", "_").replace("RandomForest", "RF").split("_")[0] for name
     in pareto_df["experiment_name"].values]
    approaches = pareto_df["approach"].unique()
    selected_metric_display = METRIC.replace("_", " ").title()
    approach_colors = {
        'SA': plt.cm.coolwarm(0.2),
        'RUL': plt.cm.coolwarm(0.8)
    }

    edge_color="black"
    bg_color="#ffffff"
    unique_sybmols=['o','s','D','^','v','P','X','*','h','8','<','>','p','H','+']
    match_sumbols={}
    for i,alg in enumerate(sorted(pareto_df['algorithm'].unique())):
        match_sumbols[alg]=unique_sybmols[i%len(unique_sybmols)]

    from paretoset import paretoset

    if len(pareto_df) > 1:
        pareto_data = pareto_df[['duration', 'metric_value']].values
        pareto_data_transformed = np.column_stack([
            pareto_data[:, 0],  # duration (minimize)
            pareto_data[:, 1]  # negative metric (maximize)
        ])

        mask = paretoset(pareto_data_transformed, sense=["min", "min"])
        pareto_df['is_pareto'] = mask
    else:
        pareto_df['is_pareto'] = True

    # ax.set_facecolor('#1B2C3D')

    # Plot non-Pareto points
    non_pareto = pareto_df[~pareto_df['is_pareto']]
    if len(non_pareto) > 0:
        for approach in approaches:
            approach_data = non_pareto[non_pareto['approach'] == approach]
            for algorithm in approach_data['algorithm'].unique():
                algo_data = approach_data[approach_data['algorithm'] == algorithm]
                if not algo_data.empty:
                    ax.scatter(algo_data['duration'], algo_data['metric_value'],
                               c=approach_colors.get(approach, '#275CE6'), s=150,
                               edgecolors=edge_color, linewidths=2, #alpha=0.6,
                               # label=f'{approach} (Non-Pareto)',
                               label=f'{algorithm}',
                               marker=match_sumbols[algorithm],
                               zorder=2)

    # Plot Pareto optimal points
    pareto_optimal = pareto_df[pareto_df['is_pareto']].sort_values('duration')
    if len(pareto_optimal) > 0:
        for approach in approaches:
            approach_data = pareto_optimal[pareto_optimal['approach'] == approach]
            for algorithm in approach_data['algorithm'].unique():
                algo_data = approach_data[approach_data['algorithm'] == algorithm]
                if not algo_data.empty:
                    ax.scatter(algo_data['duration'], algo_data['metric_value'],
                               c=approach_colors.get(approach, '#275CE6'), s=200,
                               edgecolors=edge_color, linewidths=2,
                               label=f'{algorithm}', zorder=3, marker=match_sumbols[algorithm])

        # Draw Pareto front line
        if len(pareto_optimal) > 1:
            ax.plot(pareto_optimal['duration'], pareto_optimal['metric_value'],
                    color=edge_color, linewidth=2, linestyle='--',
                    alpha=0.7, zorder=1, label='Pareto Front')

    # Annotate all points
    changepoint=1
    for idx, row in pareto_df.iterrows():
        changepoint=changepoint*-1
        label = f"{row['algorithm']}"
        # ax.annotate(label,
        #             (row['duration'], row['metric_value']),
        #             xytext=(8, 8), textcoords='offset points',
        #             fontsize=8, color=edge_color, fontweight='bold',
        #             bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color,
        #                       edgecolor=edge_color, alpha=0.8))

    # ax.set_xlabel('Duration (seconds)', fontsize=12, fontweight='bold', color=edge_color)
    # ax.set_ylabel(f'{selected_metric_display}',
    #               fontsize=12, fontweight='bold', color=edge_color)


    # Set x-axis to logarithmic scale
    ax.set_xscale('log')

    ax.tick_params(axis='x', colors=edge_color, labelsize=10)
    ax.tick_params(axis='y', colors=edge_color, labelsize=10)
    if legend_:
        # Add legend
        legend = ax.legend(loc='lower left',bbox_to_anchor=(0, 1.2), fontsize=12, framealpha=0.9, ncol=6)
        legend.get_frame().set_facecolor(bg_color)
        legend.get_frame().set_edgecolor(edge_color)
        for text in legend.get_texts():
            text.set_color(edge_color)

    ax.grid(alpha=0.2, linestyle='--', color=edge_color)
    ax.set_axisbelow(True)

def pareto():

    METRIC="mdape"
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 6),sharey="row")
    # pareto_inner_2(METRIC, ax, dff, legend_=False)
    pareto_inner("mdape", ax[0][0], filename1="Results/mlflow_Azure_SA.csv", filename2="Results/mlflow_Azure_RUL.csv",legend_=True)
    pareto_inner("IBS", ax[1][0], filename1="Results/mlflow_Azure_SA.csv", filename2="Results/mlflow_Azure_RUL.csv",legend_=False)
    ax[0][0].set_ylabel("MdAPE", fontsize=14)
    ax[1][0].set_ylabel("IBS", fontsize=14)

    # ax[1][0].set_xlabel("Total runtime", fontsize=14)
    ax[1][1].set_xlabel("Total runtime (s)", fontsize=14)
    # ax[1][2].set_xlabel("Total runtime", fontsize=14)


    pareto_inner("mdape", ax[0][1], filename1="Results/mlflow_SCANIA_SA.csv", filename2="Results/mlflow_SCANIA_RUL.csv", legend_=False)
    pareto_inner("IBS", ax[1][1], filename1="Results/mlflow_SCANIA_SA.csv", filename2="Results/mlflow_SCANIA_RUL.csv",legend_=False)

    pareto_inner("mdape", ax[0][2], filename1="Results/mlflow_HNEI_SA.csv", filename2="Results/mlflow_HNEI_RUL.csv",legend_=False)
    pareto_inner("IBS", ax[1][2], filename1="Results/mlflow_HNEI_SA.csv", filename2="Results/mlflow_HNEI_RUL.csv", legend_=False)

    ax[0][0].set_title("AZURE", fontsize=16)
    ax[0][1].set_title("HNEI", fontsize=16)
    ax[0][2].set_title("SCANIA", fontsize=16)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mlflow_port = 5011
    run_mlflow_server(mlflow_port=mlflow_port)
    mlflow.set_tracking_uri(f"http://127.0.0.1:{mlflow_port}")
    args = parse_args()

    if args.plot == "global":
        figure_global()

    elif args.plot == "calibration":
        calibrated_threshold_analysis()

    elif args.plot == "bins":
        plot_for_initial_bins(mlflow_port)

    elif args.plot == "datasets":
        datasets_plots()

    elif args.plot == "hm_sigmoid":
        HM_vs_Sigmoid()
    elif args.plot == "family_box":
        per_Category()
    elif args.plot == "latex_table":
        general_latex_table()
    elif args.plot == "censored_HNEI":
        datasets_plots_cens()

    # pareto()
    # runtimeplot_comparison()
    # datasets_plots_cens()
    # per_Category()
    # general_latex_table()
    # figure_global() # Performance of predictive models in MBS, IBS, MdAPE and MAPE in AZURE and SCANIA dataset plot
    # calibrated_threshold_analysis() # Analysis of calibrated thresholding to derive RUL prediction from ISD plot
    # plot_for_initial_bins(mlflow_port) # MdAPE per RUL bin analysis plot for Azure and SCANIA datasets
    # datasets_plots()  #dataset label information's
    # HM_vs_Sigmoid() # Statistical comparison between Hard-mapping and Sigmoid-based for translating RUL to ISD.
