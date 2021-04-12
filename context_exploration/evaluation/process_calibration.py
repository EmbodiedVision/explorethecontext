"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from context_exploration.model.loader import get_run_directory, load_config


def aggregate_calibration_data(df, aggregation):
    def q20(x):
        return x.quantile(0.2)

    def q80(x):
        return x.quantile(0.8)

    def get_aggregation(agg):
        if agg == "mean":
            agg_df = df.groupby(["obs_idx"]).mean()
        elif agg == "std":
            agg_df = df.groupby(["obs_idx"]).std()
        elif agg == "min":
            agg_df = df.groupby(["obs_idx"]).min()
        elif agg == "max":
            agg_df = df.groupby(["obs_idx"]).max()
        elif agg == "q20":
            agg_df = df.groupby(["obs_idx"]).agg(q20)
        elif agg == "q80":
            agg_df = df.groupby(["obs_idx"]).agg(q80)
        elif agg == "median":
            agg_df = df.groupby(["obs_idx"]).median()
        else:
            raise ValueError

        if "mse_openloop" in agg_df.columns:
            agg_df = agg_df[["mse_random", "mse_mpc", "mse_openloop"]]
        else:
            agg_df = agg_df[["mse_random", "mse_mpc"]]

        agg_df = agg_df.rename(
            columns={
                "mse_random": "Random calibration",
                "mse_mpc": "MPC calibration",
                "mse_openloop": "OL calibration",
            }
        )
        agg_df.index.names = ["prediction horizon"]
        agg_df = agg_df.drop([0])  # drop "0" prediction horizon (reconstruction)
        return agg_df

    mean_df = get_aggregation(aggregation)
    min_df = get_aggregation("q20")
    max_df = get_aggregation("q80")
    mean_arr, min_arr, max_arr = (
        mean_df.to_numpy(),
        min_df.to_numpy(),
        max_df.to_numpy(),
    )
    assert np.all(min_arr < max_arr)
    return mean_arr, min_arr, max_arr


def load_calibration_data(run_dict):
    if "id" in run_dict:
        if "id_list" in run_dict:
            raise ValueError("Can only set id or id_list")
        df = load_calibration_data_single_run(run_dict["id"], run_dict)
    elif "id_list" in run_dict:
        df_list = [
            load_calibration_data_single_run(id_, run_dict)
            for id_ in run_dict["id_list"]
        ]
        df = pd.concat(df_list)
    else:
        raise ValueError("Neither id nor id_list defined")
    return df


def load_calibration_data_single_run(run_id, run_dict):
    criterion = run_dict["criterion"]
    checkpoint_step = run_dict["checkpoint_step"]
    rollout_length = run_dict["rollout_length"]
    n_env_inits = run_dict["n_env_inits"]
    cem_planning_horizon = run_dict.get("cem_planning_horizon", 20)
    run_directory = get_run_directory(run_id)
    config = load_config(run_id)
    eval_df_filename = (
        f"evaluation_df_mpc_"
        f"{criterion}_"
        f"step{checkpoint_step}_"
        f"length{rollout_length}_"
        f"nenvinits{n_env_inits}_"
        f"{f'cemhorizon{cem_planning_horizon}_' if cem_planning_horizon != 20 else ''}"
        f"nctx50.pkl"
    )
    eval_df_file = run_directory.joinpath(eval_df_filename)
    eval_df = pd.read_pickle(eval_df_file)
    return eval_df


def plot_calibration_results(
    run_dict_list,
    fig_scale=1,
    aggregation="median",
    legend=True,
    legend_kwargs=None,
    ax=None,
):
    """
    Plot calibration results

    Parameters
    ----------
    run_dict_list : list[dict]
        List of dicts, each with keys
            id: Run id
            label: Legend label
            plot_opts: Opts for plotting
            criterion: Calibration criterion (e.g., neg_entropy)
            checkpoint_step: Checkpoint step
    """

    if ax is None:
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(5.4 * 0.5 * fig_scale, 1.5 * fig_scale)
        )
    for run in run_dict_list:
        eval_df = load_calibration_data(run)
        mean_arr, min_arr, max_arr = aggregate_calibration_data(eval_df, aggregation)

        x = np.arange(1, mean_arr.shape[0] + 1)
        if run["label"] != "":
            label = run["label"] + ", "
        else:
            label = ""

        if "has_minmax" not in run or run["has_minmax"] is True:
            run["has_minmax"] = ["rand", "ol", "mpc"]
        if not run["has_minmax"]:
            run["has_minmax"] = []

        if run.get("has_rand", True):
            l = ax.plot(x, mean_arr[:, 0], label=label + "Random")
            if "rand" in run["has_minmax"]:
                ax.fill_between(
                    x, min_arr[:, 0], max_arr[:, 0], alpha=0.15, color=l[0].get_color()
                )

        if run.get("has_ol", True):
            l = ax.plot(x, mean_arr[:, 2], label=label + "OL")
            if "ol" in run["has_minmax"]:
                ax.fill_between(
                    x, min_arr[:, 2], max_arr[:, 2], alpha=0.15, color=l[0].get_color()
                )

        if run.get("has_mpc", True):
            l = ax.plot(x, mean_arr[:, 1], label=label + "MPC")
            if "mpc" in run["has_minmax"]:
                ax.fill_between(
                    x, min_arr[:, 1], max_arr[:, 1], alpha=0.15, color=l[0].get_color()
                )

        ax.set_xlabel("Prediction horizon")
        ax.set_ylabel("Mean squared error")
        lower_bar = mean_arr - min_arr
        upper_bar = max_arr - mean_arr

    ax.set(yscale="log")
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    if legend:
        if legend_kwargs is None:
            legend_kwargs = {}
        plt.legend(handlelength=1, columnspacing=0.5, labelspacing=0.1, **legend_kwargs)
