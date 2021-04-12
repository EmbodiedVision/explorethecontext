"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from context_exploration.data.envs import make_env
from context_exploration.evaluation.calibration import (
    optimize_action_sequence_mpc_multistart,
    optimize_action_sequence_openloop_multistart,
)
from context_exploration.evaluation.evaluation_helpers import (
    generate_context_set,
    generate_prediction_rollout,
    generate_rollout,
)
from context_exploration.model.context_encoder import ContextSet
from context_exploration.model.loader import get_run_directory, load_model


def generate_validation_rollouts(env, n_rollouts, n_transitions, base_seed):
    validation_rollouts = []
    rollout_seed_gen = np.random.RandomState(base_seed)
    for rollout_idx in range(n_rollouts):
        rollout_seed = rollout_seed_gen.randint(0, int(1e8))
        rollout = generate_rollout(
            env,
            env_seed=rollout_seed,
            actions=env.excitation_controller.get_iterator(
                excitation_seed=rollout_seed
            ),
            max_transitions=n_transitions,
        )
        validation_rollouts.append(rollout)
    return validation_rollouts


def evaluate_optimal_calibration(
    validation_rollouts, optimal_context_latent, transition_model
):
    prediction_rollouts = [
        generate_prediction_rollout(
            validation_rollout, transition_model, optimal_context_latent
        )
        for validation_rollout in validation_rollouts
    ]
    return assemble_mse_df(validation_rollouts, prediction_rollouts)


def assemble_mse_df(validation_rollouts, prediction_rollouts):
    prediction_observations = np.stack([r["observation"] for r in prediction_rollouts])
    validation_observations = np.stack([r["observation"] for r in validation_rollouts])
    # n_rollouts, T, state_dim
    mse = np.mean((validation_observations - prediction_observations) ** 2, axis=-1)
    # mse: n_rollouts, T
    indices = np.moveaxis(np.indices(mse.shape), 0, -1)
    # indices: T, n_rollouts, 2 [last axis: [timestep_idx, rollout_idx]
    indices_flat = indices.reshape(-1, indices.shape[-1])
    mse_flat = mse.reshape(-1, 1)
    wtype = np.dtype([("val_rollout_idx", "i4"), ("obs_idx", "i4"), ("mse", "f4")])
    w = np.empty(len(indices_flat), dtype=wtype)
    w["val_rollout_idx"] = indices_flat[:, 0]
    w["obs_idx"] = indices_flat[:, 1]
    w["mse"] = mse_flat[:, 0]
    df = pd.DataFrame.from_records(w).reset_index().drop("index", axis=1)
    df = df.set_index(["val_rollout_idx", "obs_idx"])
    return df


def generate_random_multi_reset_context_set(
    env, env_base_seed, n_env_inits, max_transitions
):
    # Generate context set from multiple environment restarts
    env_seed_gen = np.random.RandomState(env_base_seed)
    current_context_set = ContextSet.create_empty()
    calibration_rollouts = []
    assert n_env_inits > 0
    for _ in range(n_env_inits):
        initial_state_seed = env_seed_gen.randint(0, int(1e8))
        random_context_set, calibration_rollout_random = generate_context_set(
            env,
            env_seed=initial_state_seed,
            actions=env.excitation_controller.get_iterator(initial_state_seed),
            max_transitions=max_transitions,
            return_rollout=True,
        )
        calibration_rollouts.append(calibration_rollout_random)
        current_context_set += random_context_set
    return current_context_set, calibration_rollouts


def validate_mpc(
    run_id,
    checkpoint_step,
    calibration_rollout_length,
    n_env_inits,
    cem_planning_horizon,
    n_validation_transitions,
):
    device = "cuda"
    criterion = "neg-entropy"
    propagate_context = False

    run_directory = get_run_directory(run_id)

    verbose = False
    cem_kwargs = dict(
        candidates=1000,
        top_candidates=100,
        clip_actions=True,
        return_mean=True,
        verbose=verbose,
    )
    # context samples during CEM
    n_context_samples = 20

    n_contexts = 50

    n_calibrations_per_context = 1
    n_validation_rollouts_per_context = 20

    eval_df_filename = (
        f"evaluation_df_mpc_"
        f"{criterion}_"
        f"{'propctx_' if propagate_context else ''}"
        f"step{checkpoint_step}_"
        f"length{calibration_rollout_length}_"
        f"nenvinits{n_env_inits}_"
        f"{f'cemhorizon{cem_planning_horizon}_' if cem_planning_horizon != 20 else ''}"
        f"nctx{n_contexts}.pkl"
    )
    if run_directory.joinpath(eval_df_filename).is_file():
        print("This evaluation already exists, exiting.")
        return

    calib_data_filename = (
        f"calibration_data_mpc_"
        f"{criterion}_"
        f"{'propctx_' if propagate_context else ''}"
        f"step{checkpoint_step}_"
        f"length{calibration_rollout_length}_"
        f"nenvinits{n_env_inits}_"
        f"{f'cemhorizon{cem_planning_horizon}_' if cem_planning_horizon != 20 else ''}"
        f"nctx{n_contexts}.pkl"
    )

    env_name, transition_model, context_encoder, log_likelihood_model = load_model(
        run_id, checkpoint_step, device
    )
    transition_model.eval()
    context_encoder.eval()
    if log_likelihood_model:
        log_likelihood_model.eval()

    df_list = []
    calibration_rollout_list = []

    context_seed_gen = np.random.RandomState(113322)

    for context_idx in tqdm(range(n_contexts)):
        env = make_env(env_name)
        if hasattr(env, "max_duration"):
            env.max_duration = 100_000
        context_seed = context_seed_gen.randint(0, int(1e8))
        env.initialize_context(context_seed)
        validation_rollouts = generate_validation_rollouts(
            env,
            n_validation_rollouts_per_context,
            n_validation_transitions,
            base_seed=context_seed,
        )
        initial_state_seed_gen = np.random.RandomState(context_seed + 1)
        for calibration_idx in tqdm(range(n_calibrations_per_context)):
            initial_state_seed = initial_state_seed_gen.randint(0, int(1e8))
            (
                random_context_set,
                calibration_rollouts_random,
            ) = generate_random_multi_reset_context_set(
                env, initial_state_seed, n_env_inits, calibration_rollout_length
            )
            random_context_latent = context_encoder.forward_set(random_context_set)
            random_mse = evaluate_optimal_calibration(
                validation_rollouts, random_context_latent, transition_model
            )
            (
                openloop_context,
                calibration_rollouts_ol,
            ) = optimize_action_sequence_openloop_multistart(
                env,
                initial_state_seed,
                n_env_inits,
                calibration_rollout_length,
                transition_model,
                context_encoder,
                log_likelihood_model,
                criterion,
                propagate_context,
                n_context_samples,
                cem_kwargs,
                verbose=verbose,
            )
            openloop_mse = evaluate_optimal_calibration(
                validation_rollouts, openloop_context, transition_model
            )
            (
                mpc_context,
                calibration_rollouts_mpc,
            ) = optimize_action_sequence_mpc_multistart(
                env,
                initial_state_seed,
                n_env_inits,
                calibration_rollout_length,
                transition_model,
                context_encoder,
                log_likelihood_model,
                cem_planning_horizon,
                criterion,
                propagate_context,
                n_context_samples,
                cem_kwargs,
                verbose=verbose,
            )
            mpc_mse = evaluate_optimal_calibration(
                validation_rollouts, mpc_context, transition_model
            )
            df = pd.concat(
                (
                    openloop_mse.rename(columns={"mse": "mse_openloop"}),
                    random_mse.rename(columns={"mse": "mse_random"}),
                    mpc_mse.rename(columns={"mse": "mse_mpc"}),
                ),
                axis=1,
            ).reset_index()
            df["calibration_rollout_length"] = calibration_rollout_length
            df["calibration_idx"] = calibration_idx
            df["context_idx"] = context_idx
            df_list.append(df)
            calibration_rollout_list.append(
                {
                    "context_seed": context_seed,
                    "initial_state_seed": initial_state_seed,
                    "calibration_idx": calibration_idx,
                    "context_idx": context_idx,
                    "random_context_latent": random_context_latent,
                    "mpc_context_latent": mpc_context,
                    "ol_context_latent": openloop_context,
                    "calibration_rollout_random": calibration_rollouts_random,
                    "calibration_rollout_mpc": calibration_rollouts_mpc,
                    "calibration_rollout_ol": calibration_rollouts_ol,
                }
            )

        env.release_context()

    full_df = pd.concat(df_list)
    full_df.to_pickle(run_directory.joinpath(eval_df_filename))
    with open(run_directory.joinpath(calib_data_filename), "wb") as handle:
        pickle.dump(calibration_rollout_list, handle)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run calibration")
    parser.add_argument("run_id", type=str, help="Run id")
    parser.add_argument("--ckpt", type=str, help="Checkpoint")
    parser.add_argument("--rlength", type=int, help="Calibration rollout length")
    parser.add_argument("--nenvinits", type=int, help="Number of env. inits")
    parser.add_argument("--cemph", type=int, default=20, help="CEM planning horizon")
    parser.add_argument(
        "--vlength", type=int, default=50, help="Validation rollout length"
    )
    args = parser.parse_args()

    run_id = args.run_id
    checkpoint_step = args.ckpt
    calibration_rollout_length = args.rlength
    n_env_inits = args.nenvinits
    cem_planning_horizon = args.cemph
    n_validation_transitions = args.vlength

    with torch.no_grad():
        validate_mpc(
            run_id,
            checkpoint_step,
            calibration_rollout_length,
            n_env_inits,
            cem_planning_horizon,
            n_validation_transitions,
        )


if __name__ == "__main__":
    main()
