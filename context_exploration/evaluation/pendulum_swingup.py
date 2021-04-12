"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from context_exploration.cem.cem import CEM
from context_exploration.data.envs import make_env
from context_exploration.model.loader import get_run_directory, load_model


class InitialCemState:
    def __init__(self, state):
        self.state = state

    @property
    def device(self):
        return self.state.device

    @property
    def batch_shape(self):
        return self.state.shape[:-1]

    def expand(self, target_shape):
        target_shape = torch.Size(target_shape)
        return InitialCemState(
            self.state[..., None, :].expand(*target_shape, self.state.shape[-1])
        )


class CemTransitionModel:
    def __init__(self, transition_model, context_latent):
        """
        Transition model for CEM

        Parameters
        ----------
        transition_model : TransitionModel
        context_latent : torch.Tensor, [B x context_dim]
        """
        self.transition_model = transition_model
        self.action_size = transition_model.action_dim
        self.context_latent = context_latent

    def multi_step(self, initial_cem_state, action_sequence):
        """
        Multi-step transition model

        Parameters
        ----------
        initial_cem_state : InitialCemState, shape [B x n_candidates x state_dim]
        action_sequence : torch.Tensor, [T x B x n_candidates x action_dim]

        Returns
        -------
        predicted_states: torch.Tensor,
            [(T+1) x B x n_candidates x state_dim]
        """
        ctx_mean = self.context_latent.mean
        state = initial_cem_state.state
        ctx_mean = ctx_mean[:, None, :].expand(
            ctx_mean.shape[0], state.shape[1], ctx_mean.shape[-1]
        )
        prediction = self.transition_model.forward_multi_step(
            state, action_sequence, ctx_mean, return_mean_only=True
        )
        return prediction


class PendulumCemReturnModel:
    def __init__(self):
        """
        Return model for CEM
        """
        pass

    def forward(self, predicted_states, actions):
        """
        Compute 'return' for pendulum swingup task

        Parameters
        ----------
        predicted_states: torch.Tensor,
            [(T+1) x B x n_candidates x state_dim]
            Includes initial state
        actions: torch.Tensor,
            [T x B x n_candidates x state_dim]

        Returns
        -------
        return_: torch.Tensor, [B x n_candidates]
        """
        assert actions.dim() == 4
        assert actions.shape[-1] == 1
        assert predicted_states.dim() == 4
        device = predicted_states.device
        actions = actions.squeeze(-1).cpu().numpy()
        predicted_states = predicted_states.cpu().numpy()
        # the first cost we compute is for action[0] applied to predicted_states[0]
        predicted_states = predicted_states[:-1]
        cos_th = predicted_states[..., 0]
        sin_th = predicted_states[..., 1]
        cos_th = np.clip(cos_th, -1, 1)
        sin_th = np.clip(sin_th, -1, 1)
        actions = np.clip(actions, -2, 2)
        thdot = predicted_states[..., 2]
        th = np.arctan2(sin_th, cos_th)
        norm_angle = ((th + np.pi) % (2 * np.pi)) - np.pi
        costs = norm_angle ** 2 + 0.1 * thdot ** 2 + 0.001 * (actions ** 2)
        return_ = -torch.Tensor(costs).float().to(device).sum(dim=0).unsqueeze(-1)
        return return_


def run_swingup(
    env_name,
    context_seed,
    initial_state_seed,
    transition_model,
    context_latent,
    rollout_length,
    cem_planning_horizon,
    cem_kwargs,
    render=False,
):

    env = make_env(env_name)
    env.initialize_context(context_seed)
    env.seed(initial_state_seed)
    state = env.reset(init_mode="calibration")

    cem_transition_model = CemTransitionModel(transition_model, context_latent)
    cem_return_model = PendulumCemReturnModel()

    device = transition_model.device

    cem = CEM(
        cem_transition_model,
        cem_return_model,
        planning_horizon=cem_planning_horizon,
        action_space=env.action_space,
        return_all_actions=True,
        **cem_kwargs,
    )

    total_reward = 0

    for step_idx in range(rollout_length):
        # introduce batch dimension
        cem_state = InitialCemState(torch.Tensor(state).to(device).float().unsqueeze(0))
        optimal_action_sequence, _ = cem.forward(cem_state)
        action = optimal_action_sequence[0][0].cpu().numpy()
        # apply first action, remove batch
        state, reward, done, info = env.step(action)
        total_reward += reward
        if render:
            env.render(mode="human")

    env.close()

    swingup_data = {
        "context_seed": context_seed,
        "initial_state_seed": initial_state_seed,
        "total_reward": total_reward,
    }

    return swingup_data


def main():
    run_id = "cr_s3_pendulum_bd_posweights_relu_npklw5"
    device = "cuda"
    criterion = "neg-entropy"
    checkpoint_step = "100000_best"
    calibration_rollout_length = 30
    calibration_cem_ph = 20
    n_env_inits = 1

    run_directory = get_run_directory(run_id)
    env_name, transition_model, context_encoder, log_likelihood_model = load_model(
        run_id, checkpoint_step, device
    )

    calib_data_filename = (
        f"calibration_data_mpc_"
        f"{criterion}_"
        f"step{checkpoint_step}_"
        f"length{calibration_rollout_length}_"
        f"nenvinits{n_env_inits}_"
        f"{f'cemhorizon{calibration_cem_ph}_' if calibration_cem_ph != 20 else ''}"
        f"nctx50.pkl"
    )
    with open(run_directory.joinpath(calib_data_filename), "rb") as handle:
        calib_data = pickle.load(handle)

    verbose = False
    cem_kwargs = dict(
        candidates=1000,
        top_candidates=100,
        clip_actions=True,
        return_mean=True,
        verbose=verbose,
    )
    cem_planning_horizon = 20
    rollout_length = 50

    print(f"There are {len(calib_data)} calibration rollouts")

    swingup_data_list = []

    for calibration in tqdm(calib_data):
        context_seed = calibration["context_seed"]
        seed_gen = np.random.RandomState(context_seed + 113344)
        # 1 swingup trial per calibration
        for initial_state_index in tqdm(range(1)):
            # initial_state_seed = calibration["initial_state_seed"]
            initial_state_seed = seed_gen.randint(0, int(1e8))
            for calib_type in ["mpc", "random"]:
                context_latent = calibration[calib_type + "_context_latent"]
                with torch.no_grad():
                    swingup_data = run_swingup(
                        env_name,
                        context_seed,
                        initial_state_seed,
                        transition_model,
                        context_latent,
                        rollout_length,
                        cem_planning_horizon,
                        cem_kwargs,
                        render=False,
                    )
                    swingup_data["calib_type"] = calib_type
                    swingup_data_list.append(swingup_data)

    swingup_data_df = pd.DataFrame(swingup_data_list)

    swingup_data_filename = (
        f"swingup_data_mpc_"
        f"{criterion}_"
        f"step{checkpoint_step}_"
        f"length{calibration_rollout_length}_"
        f"nenvinits{n_env_inits}_"
        f"nctx50.pkl"
    )
    swingup_data_df.to_pickle(run_directory.joinpath(swingup_data_filename))


if __name__ == "__main__":
    main()
