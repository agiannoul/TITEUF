import mlflow
import numpy as np
import pandas as pd


def get_run_ids_zero(datasetname="SCANIA", SA_or_RUL="SA"):
    from mlflow.tracking import MlflowClient
    import math

    client = MlflowClient()

    # Get experiments ending in _TEST
    test_experiments = client.search_experiments(
        filter_string=f"name LIKE '%{SA_or_RUL}% {datasetname}%'"
    )
    test_experiments2 = client.search_experiments(
        filter_string=f"name LIKE '%{SA_or_RUL}% {datasetname.upper()}%'"
    )
    exps_ids=[exp.experiment_id for exp in test_experiments]
    test_experiments2=[exp for exp in test_experiments2 if exp.experiment_id not in exps_ids]
    test_experiments=test_experiments+test_experiments2
    print(len(test_experiments))
    rows = []
    all_metric_keys = set()

    for exp in test_experiments:
        if "Train-Val" in exp.name or "TEST" in exp.name:
            continue
        if exp.name.replace("_TEST", "").replace("My RUL experiment ", "").replace(
                "My SA experiment ", "").replace(SA_or_RUL, "").split(" ")[-1] in [rr["method"] for rr in rows]:
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
            "method": exp.name.split(" ")[-1],
            "run_id": best_run.info.run_id,
            **new_metrics
        }
        rows.append(row)
    return rows


def get_run_ids( SA_or_RUL="SA"):
    from mlflow.tracking import MlflowClient
    import math

    client = MlflowClient()

    # Get experiments ending in _TEST
    test_experiments = client.search_experiments(
        filter_string=f"name LIKE '%CENS_H_% %'"
    )

    print(len(test_experiments))
    rows = []
    all_metric_keys = set()

    for exp in test_experiments:
        if "Train-Val" in exp.name:
            continue


        # Search runs
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
        )

        # ---------------------------------------------
        # Keep only runs where "IBR" metric exists and is NOT NaN
        # ---------------------------------------------
        try:
            valid_runs = []
            for r in runs:
                metrics = r.data.metrics
                if "IBS" in metrics and metrics["IBS"] is not None and not math.isnan(metrics["IBS"]):
                    valid_runs.append(r)

            # ---------------------------------------------
            # Select the most recent run (by start time)
            # ---------------------------------------------
            valid_runs.sort(key=lambda x: x.info.start_time, reverse=True)
            best_run = valid_runs[0]
        except Exception as e:
            print(f"Skipping experiment {exp.name} due to error: {e}")
            continue
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
        flavor = exp.name.split(" ")[0]
        method = exp.name.split(" ")[-1]
        porpotion = exp.name.split("_H_")[1][0]
        seed = exp.name.split("seed")[1][0]
        row = {
            "flavor": flavor,
            "method": method,
            "proportion": porpotion,
            "seed": seed,
            "run_id": best_run.info.run_id,
            **new_metrics
        }
        rows.append(row)
    return rows

import matplotlib.pyplot as plt

def plot_metric_vs_proportion(result_df, metric):
    """
    Plots the given metric for all methods as the proportion changes.

    Args:
        result_df (pd.DataFrame): DataFrame with columns 'method', 'proportion', and metric columns.
        metric (str): The metric column to plot.
    """
    plt.figure(figsize=(8, 6))
    for method in result_df["method"].unique():
        df_method = result_df[result_df["method"] == method]
        df_method["proportion"] = df_method["proportion"].astype(float)
        df_method.sort_values(by="proportion", inplace=True)
        plt.plot(df_method["proportion"], df_method[metric], marker="o", label=method)
    plt.xlabel("Proportion")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Proportion for all Methods")
    plt.legend()
    plt.tight_layout()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_censoring_heatmap(ax, df, metric_col="metric",
                           method_col="method", prop_col="proportion",
                           baseline_prop=0.0, cmap="coolwarm", center=0,
                           title=None, cbar=True, fmt=".2f", sorted_columns=None, second=False,cbar_ax=None,):
    """
    Heatmap of relative performance drop vs baseline.

    Each cell is split:
    - Top-left half: relative drop (colored)
    - Bottom-right half: rank + arrow (white background)

    No diagonal lines and no rank-based coloring.
    """

    # Pivot to matrix form
    pivot = df.pivot(index=method_col, columns=prop_col, values=metric_col)

    # Compute relative drop vs baseline (loop version kept)
    baseline = pivot[baseline_prop]
    for col in pivot.columns:
        pivot[col] = [max(val,zer) for zer,val in zip(pivot[pivot.columns[0]].values,pivot[col].values)]
    rel_drop = []
    for i, method in enumerate(pivot.index):
        row = []
        for j, prop in enumerate(pivot.columns):
            base = baseline[method]
            value = pivot.loc[method, prop]
            drop = abs((base - value) / base)
            row.append(drop)
        rel_drop.append(row)
    rel_drop = np.array(rel_drop)

    rel_drop_df = pd.DataFrame(rel_drop, index=pivot.index, columns=pivot.columns)

    # Sort methods using your provided ordering
    if sorted_columns is not None:
        rel_drop_df.sort_values(
            by=method_col,
            key=lambda x: x.map({k: i for i, k in enumerate(sorted_columns)}),
            inplace=True
        )

    # Reorder pivot accordingly
    pivot = pivot.loc[rel_drop_df.index]

    # Compute ranks (lower metric = better → rank 1 best)
    ranks = pivot.rank(axis=0, ascending=True, method="min")
    baseline_ranks = ranks[baseline_prop]
    rank_delta = ranks.subtract(baseline_ranks, axis=0)

    # Plot heatmap using relative drop
    sns.heatmap(
        rel_drop_df,
        ax=ax,
        vmax=1,
        cmap=cmap,
        annot=False,
        linewidths=0.25,
        cbar=cbar,
        cbar_ax=cbar_ax,
        alpha=0.6,
    )

    # Overlay white triangles + annotations (no diagonal lines)
    for i, method in enumerate(rel_drop_df.index):
        for j, prop in enumerate(rel_drop_df.columns):
            drop_val = rel_drop_df.loc[method, prop]
            rank_val = int(ranks.loc[method, prop])
            delta = rank_delta.loc[method, prop]

            # Arrow logic
            if delta < 0:
                arrow = "↓"
            elif delta > 0:
                arrow = "↑"
            else:
                arrow = "→"

            # White triangle (bottom-right half)
            triangle = Polygon(
                [(j+1, i), (j+1, i+1), (j, i+1)],
                closed=True,
                facecolor="white",
                edgecolor="none",
                zorder=3
            )
            ax.add_patch(triangle)

            # Top-left text: drop
            ax.text(j + 0.01, i + 0.01, f"{drop_val:{fmt}}",
                    ha="left", va="top", fontsize=13, color="black", zorder=5)

            # Bottom-right text: rank + arrow
            ax.text(j + 0.9, i + 0.8, f"{rank_val} {arrow}",
                    ha="right", va="bottom", fontsize=13, color="black", zorder=5)


    ax.set_xlabel("Censoring proportion")
    if not second:
        ax.set_ylabel("Method")
        ax.set_ylabel("Method", fontsize=12)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlabel("Censoring proportion", fontsize=13)

    if title:
        ax.set_title(title,fontsize=14)



if __name__ == "__main__":
    runs = get_run_ids(SA_or_RUL="RUL")
    runs2 = get_run_ids_zero(SA_or_RUL="RUL",datasetname="HNEI")
    runs3 = get_run_ids_zero(SA_or_RUL="SA",datasetname="HNEI")
    runs2.extend(runs3)
    dfzero=pd.DataFrame(runs2)
    dfzero["proportion"]=[0 for _ in range(len(dfzero))]
    dfzero["flavor"]=["RUL" for _ in range(len(dfzero))]
    dfzero["seed"]=[1 for _ in range(len(dfzero))]


    df = pd.DataFrame(runs)
    df=pd.concat([df,dfzero],ignore_index=True)
    df["proportion"]=df["proportion"].astype(float)

    columns=["flavor","method","proportion","seed","run_id"]

    result_rows = []
    for method in df["method"].unique():
        for proportion in df["proportion"].unique():
            df_method_prop = df[(df["method"] == method) & (df["proportion"] == proportion)]
            metric_cols = [col for col in df_method_prop.columns if col not in columns]
            avg_metrics = df_method_prop[metric_cols].mean()
            row = {"method": method, "proportion": proportion, **avg_metrics}
            result_rows.append(row)
    result_df = pd.DataFrame(result_rows)
    result_df["proportion"]=[int(prop)/8 for prop in result_df["proportion"]]
    # plot_metric_vs_proportion(result_df, "IBS")
    # plot_metric_vs_proportion(result_df, "Max_brier")
    # plot_metric_vs_proportion(result_df, "mdape")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [1, 1.05]})
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])  # shared colorbar axis

    sorted_columns=["XGBoost", "CatBoost_W_RUL", "RandomForestRUL", "ElasticNetRUL", "sktimeRocket",
                    "sktimeLSTMFCN", "sktimeInceptionTime", "sktimeCNN", "sktimeResNet", "sktimeFCN",
                    "CoxPH", "DeepHit", "RDSM", "RSF", "GradientBoosting"
                    ]

    result_df.sort_values(by="method", key=lambda x: x.map({k: i for i, k in enumerate(sorted_columns)}), inplace=True)
    result_df["method"] = [methodd.replace("CatBoost_W_RUL","CatBoost").replace("RUL","").replace("sktime","").replace("RandomForest","RF")
                           .replace("GradientBoosting","GB").replace("InceptionTime","Inception\nTime")

                           for methodd in result_df["method"]]
    sorted_columns_new=[methodd.replace("CatBoost_W_RUL","CatBoost").replace("RUL","").replace("sktime","").replace("RandomForest","RF")
                           .replace("GradientBoosting","GB").replace("InceptionTime","Inception\nTime")
                           for methodd in sorted_columns]


    metric="IBS"
    metric2="mdape"
    meetric2_disp="MdAPE"
    plot_censoring_heatmap(ax1, result_df, metric_col=metric,
                           method_col="method", prop_col="proportion",
                           baseline_prop=0.0, cmap="coolwarm", center=None,
                           title=f"Percentage difference/rank on {metric}", cbar=False, fmt=".2f",sorted_columns=sorted_columns_new)
    plot_censoring_heatmap(ax2, result_df, metric_col=metric2,
                           method_col="method", prop_col="proportion",
                           baseline_prop=0.0, cmap="coolwarm", center=None,
                           title=f"Percentage difference/rank on {meetric2_disp}", cbar=True, fmt=".2f", sorted_columns=sorted_columns_new,second=True,cbar_ax=cbar_ax)
    plt.tight_layout()
    plt.show()
