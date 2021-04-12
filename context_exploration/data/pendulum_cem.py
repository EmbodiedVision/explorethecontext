"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
from copy import deepcopy

import numpy as np
import torch
from gym.vector import AsyncVectorEnv

from context_exploration.cem.cem import CEM
from context_exploration.data.envs import (
    ActionRepeatWrapper,
    NoActionRendererPendulumQuadrantActionFactor,
    PendulumEnvQuadrantActionFactor,
)


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


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class PendulumTransitionModel:
    def __init__(self, env, action_repeat, action_factor):
        self.env = env
        self.action_repeat = action_repeat
        self.action_factor = action_factor

    @property
    def action_size(self):
        return 1

    def multi_step(self, initial_state_belief, actions):
        """
        Multi-step forward prediction

        Parameters
        ----------
        initial_state_belief: Belief with batch_shape [<bs>]
            Initial state belief
        actions: torch.Tensor [T x <bs> x action_size]
            Actions to apply. actions[0] is applied to initial_state_belief

        Returns
        -------
        state_array: np.ndarray
            States
        """
        cos_th, sin_th, thdot = (initial_state_belief.state[:, k] for k in range(3))
        th = np.arctan2(sin_th, cos_th)
        state_list = [np.stack([th, thdot], axis=1)]
        g, l, m, dt = self.env.g, self.env.l, self.env.m, self.env.dt
        for n in range(len(actions)):
            for _ in range(self.action_repeat):
                u = actions[n].cpu().numpy().squeeze(-1)
                u = np.clip(u, -2, 2)
                u = u * self.action_factor
                newthdot = (
                    thdot
                    + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l ** 2) * u)
                    * dt
                )
                newth = th + newthdot * dt
                newthdot = np.clip(newthdot, -self.env.max_speed, self.env.max_speed)
                th, thdot = newth, newthdot
                state_list.append(np.stack([th, thdot], axis=1))
        return np.stack(state_list)


class PendulumReturnModel:
    def __init__(self, action_repeat):
        self.action_repeat = action_repeat

    def forward(self, state_belief, action):
        """
        Pendulum return model

        Parameters
        ----------
        state_belief: np.ndarray with shape [T+1 x <bs> x state_dim]
        action: torch.Tensor [T x <bs> x action_size]
            Actions to apply. actions[0] is applied to state_belief[0]

        Returns
        -------
        return: torch.Tensor [<bs>, 1]
        """

        th, thdot = state_belief[:-1, :, 0], state_belief[:-1, :, 1]
        u = torch.repeat_interleave(action, self.action_repeat, dim=0)
        u = u.cpu().numpy().squeeze(-1)
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        reward = -costs
        return torch.Tensor(reward).sum(dim=0).unsqueeze(-1)


def main():
    action_factor = 1.25
    n_cem_candidates = 1000
    pendulum_env = NoActionRendererPendulumQuadrantActionFactor(
        quadrant_action_factor=np.array(
            [action_factor, action_factor, action_factor, action_factor]
        ),
        init_type="bottom",
    )
    env = ActionRepeatWrapper(pendulum_env, action_repeat=2)

    device = "cpu"

    cem_transition_model = PendulumTransitionModel(
        pendulum_env, action_repeat=2, action_factor=action_factor
    )
    cem_return_model = PendulumReturnModel(action_repeat=2)

    cem_kwargs = {"candidates": n_cem_candidates, "optimisation_iters": 10}
    cem = CEM(
        cem_transition_model,
        cem_return_model,
        planning_horizon=10,
        action_space=env.action_space,
        return_all_actions=True,
        verbose=False,
        **cem_kwargs
    )

    env.seed(42)
    obs = env.reset()
    total_reward = 0
    for step in range(50):
        initial_state = torch.from_numpy(obs).float().to(device)
        cem_initial_state = InitialCemState(initial_state)
        optimal_actions, _ = cem.forward(cem_initial_state)
        action = optimal_actions[0]
        obs, reward, done, info = env.step(action)
        print(action, obs, reward)
        total_reward += reward
        env.render(mode="human")
    print(total_reward)
    env.close()


if __name__ == "__main__":
    main()
