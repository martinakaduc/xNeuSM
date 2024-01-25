import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tueplots import bundles

plt.rcParams.update(bundles.icml2022())
sns.set_style("darkgrid", rc=bundles.icml2022())

runtime_folder = "./runtime"
performance_folder = "./performance"
degradation_folder = "./degradation"

if __name__ == "__main__":
    ##############################################################################
    sns.set(font_scale=0.8)

    # Load average runtime
    runtime = pd.read_csv(os.path.join(runtime_folder, "average.csv"))

    # Plot the runtime
    runtime_df = {"algo": [], "dataset": [], "runtime": []}
    for i, row in enumerate(runtime["algo"]):
        for dataset in runtime.columns[1:]:
            runtime_df["algo"].append(row)
            runtime_df["dataset"].append(dataset)
            runtime_df["runtime"].append(runtime[dataset][i])

    runtime_df = pd.DataFrame(runtime_df)

    fig, ax = plt.subplots(figsize=(7, 3))
    sns.barplot(data=runtime_df, x="dataset", y="runtime", hue="algo", ax=ax)
    ax.set_ylabel("Runtime $(s)$")
    ax.set_xlabel("")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(runtime_folder, "runtime_average.pdf"))
    ##############################################################################
    # Load each runtime results
    list_runtime_files = os.listdir(runtime_folder)
    list_runtime_files = [
        file
        for file in list_runtime_files
        if file.endswith(".csv") and file != "average.csv"
    ]
    setting_map = {
        "<20": "$|V_{\mathcal{P}}| \in [2, 20]$",
        "20-40": "$|V_{\mathcal{P}}| \in (20, 40]$",
        "40-60": "$|V_{\mathcal{P}}| \in (40, 60]$",
        ">60": "$|V_{\mathcal{P}}| \in (60, |V_{\mathcal{T}}|]$",
    }
    list_df = {}

    for dataset in ["KKI", "COX2", "DBLP-v1", "COX2_MD", "MSRC-21", "DHFR"]:
        # Find the runtime file starting with the dataset name
        files = [f"{dataset}-dense.csv", f"{dataset}-nondense.csv"]
        files = [file for file in list_runtime_files if file in files]
        files.sort(key=lambda x: x.split("-")[-1], reverse=True)
        num_subplots = len(files)
        # Delete the runtime file starting with the dataset name
        list_runtime_files = [file for file in list_runtime_files if file not in files]

        for file in files:
            df_name = (
                file[: file.rfind("-")]
                + " - "
                + ("Sparse queries" if "non" in file else "Dense queries")
            )
            df = pd.read_csv(os.path.join(runtime_folder, file))

            runtime_df = {"algo": [], "settings": [], "runtime": []}

            for i, row in enumerate(df["algo"]):
                for setting in df.columns[1:]:
                    runtime_df["algo"].append(row)
                    # Check runtime is nan or not
                    if np.isnan(df[setting][i]):
                        runtime_df["settings"][-1] = (
                            runtime_df["settings"][-1][:-4] + "|V_{\mathcal{T}}|]$"
                        )

                    runtime_df["settings"].append(setting_map[setting])
                    runtime_df["runtime"].append(df[setting][i])

            runtime_df = pd.DataFrame(runtime_df)
            list_df[df_name] = runtime_df

    # Draw line plot
    # Header: algo, <20, 20-40, 40-60, >60
    fig, axes = plt.subplots(figsize=(6 * 3, 3 * 3), ncols=3, nrows=3)

    # 3x3 plot
    idx = 0
    for name, df in list_df.items():
        sns.lineplot(
            data=df,
            x="settings",
            y="runtime",
            hue="algo",
            ax=axes[idx // 3][idx % 3],
            marker="o",
        )
        axes[idx // 3][idx % 3].tick_params(axis="both", which="major", labelsize=14)
        # axes[idx].tick_params(axis='x', rotation=10)
        axes[idx // 3][idx % 3].set_xlabel("", fontsize=14)
        axes[idx // 3][idx % 3].set_ylabel("Runtime $(s)$", fontsize=14)
        axes[idx // 3][idx % 3].set_yscale("log")
        axes[idx // 3][idx % 3].set_title(name, fontsize=16)
        axes[idx // 3][idx % 3].get_legend().set_visible(False)
        idx += 1
    handles, labels = axes[0][0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=9,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(runtime_folder, f"runtime_by_dataset.pdf"))
    ##############################################################################
    # Load each performance results
    list_performance_files = os.listdir(performance_folder)
    list_performance_files = [
        "KKI.csv",
        "COX2.csv",
        "COX2_MD.csv",
        "DHFR.csv",
        "DBLP-v1.csv",
        "MSRC-21.csv",
    ]

    performance_df = {
        "algo": [],
        "dataset": [],
        "ROC AUC": [],
        "PR AUC": [],
        "F1 score": [],
        "Accuracy": [],
    }
    for file in list_performance_files:
        dataset_name = file[: file.rfind(".csv")]
        df = pd.read_csv(os.path.join(performance_folder, file))
        for i, row in enumerate(df["algo"]):
            performance_df["algo"].append(row)
            performance_df["dataset"].append(dataset_name)
            performance_df["ROC AUC"].append(df["ROC AUC"][i])
            performance_df["PR AUC"].append(df["PR AUC"][i])
            performance_df["F1 score"].append(df["F1 score"][i])
            performance_df["Accuracy"].append(df["Accuracy"][i])

    performance_df = pd.DataFrame(performance_df)

    # 1x4 plot
    fig, axes = plt.subplots(figsize=(5 * 4, 4), ncols=4)
    for idx, metric in enumerate(["ROC AUC", "PR AUC", "F1 score", "Accuracy"]):
        sns.barplot(
            data=performance_df, x="dataset", y=metric, hue="algo", ax=axes[idx]
        )
        axes[idx].tick_params(axis="both", which="major", labelsize=14)
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].set_ylabel(metric, fontsize=14)
        axes[idx].set_xlabel("")
        axes[idx].set_ylim(0.5, 1)
        axes[idx].get_legend().set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(performance_folder, "performance_1x4.pdf"))

    # 2x2 plot
    fig, axes = plt.subplots(figsize=(8 * 2, 4 * 2), ncols=2, nrows=2)
    for idx, metric in enumerate(["ROC AUC", "PR AUC", "F1 score", "Accuracy"]):
        sns.barplot(
            data=performance_df,
            x="dataset",
            y=metric,
            hue="algo",
            ax=axes[idx // 2][idx % 2],
        )
        axes[idx // 2][idx % 2].tick_params(axis="both", which="major", labelsize=14)
        axes[idx // 2][idx % 2].tick_params(axis="x")
        axes[idx // 2][idx % 2].set_ylabel(metric, fontsize=14)
        axes[idx // 2][idx % 2].set_xlabel("")
        axes[idx // 2][idx % 2].set_ylim(0.5, 1)
        axes[idx // 2][idx % 2].get_legend().set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(performance_folder, "performance_2x2.pdf"))

    ##############################################################################
    # Load degradation results
    list_degradation_files = os.listdir(degradation_folder)
    list_degradation_files = [
        file
        for file in list_degradation_files
        if file.endswith(".csv") and file != "average.csv"
    ]

    list_df = {}
    for file in list_degradation_files:
        metric = file[: file.rfind(".csv")].replace("_", " ")
        df = pd.read_csv(os.path.join(degradation_folder, file))
        list_df[metric] = df

    degradation_df = {
        "dataset": [],
        "threshold": [],
        "ROC AUC": [],
        "PR AUC": [],
        "F1 score": [],
        "Accuracy": [],
    }
    for idx, row in enumerate(list_df["Accuracy"]["Threshold"]):
        for dataset in list_df["Accuracy"].columns[1:]:
            degradation_df["dataset"].append(dataset)
            degradation_df["threshold"].append(row)
            for metric, metric_df in list_df.items():
                degradation_df[metric].append(metric_df[dataset][idx])

    degradation_df = pd.DataFrame(degradation_df)

    # 2x2 plot
    fig, axes = plt.subplots(figsize=(6 * 2, 3 * 2), ncols=2, nrows=2)
    for idx, metric in enumerate(["ROC AUC", "PR AUC", "F1 score", "Accuracy"]):
        row_idx = idx // 2
        col_idx = idx % 2
        sns.lineplot(
            data=degradation_df,
            x="threshold",
            y=metric,
            hue="dataset",
            ax=axes[row_idx][col_idx],
            marker="o",
        )
        axes[row_idx][col_idx].tick_params(axis="both", which="major", labelsize=14)
        axes[row_idx][col_idx].set_xlabel("Threshold", fontsize=14)
        axes[row_idx][col_idx].set_ylabel(metric, fontsize=14)
        axes[row_idx][col_idx].set_ylim(0.5, 1)
        axes[row_idx][col_idx].get_legend().set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=6,
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(degradation_folder, "degradation_2x2.pdf"))

    # 1x4 plot
    fig, axes = plt.subplots(figsize=(5 * 4, 3), ncols=4)
    for idx, metric in enumerate(["ROC AUC", "PR AUC", "F1 score", "Accuracy"]):
        sns.lineplot(
            data=degradation_df,
            x="threshold",
            y=metric,
            hue="dataset",
            ax=axes[idx],
            marker="o",
        )
        axes[idx].tick_params(axis="both", which="major", labelsize=14)
        axes[idx].set_xlabel("Threshold", fontsize=14)
        axes[idx].set_ylabel(metric, fontsize=14)
        axes[idx].set_ylim(0.5, 1)
        axes[idx].get_legend().set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=6,
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(degradation_folder, "degradation_1x4.pdf"))
