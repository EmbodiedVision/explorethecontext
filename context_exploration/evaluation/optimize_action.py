"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import os

import matplotlib.pyplot as plt
import torch

from context_exploration.data.envs import make_env
from context_exploration.model.loader import get_run_directory, load_model


def average_entropy_plot(
    env, initial_state, transition_model, context_encoder, device, ax, label, rel
):
    initial_state, action, next_state, assignments, batchsize = generate_context_set(
        env, initial_state, transition_model, context_encoder, device
    )
    average_entropy = compute_average_entropy(
        context_encoder, initial_state, action, next_state, assignments, batchsize
    )
    average_entropy = average_entropy.cpu().numpy()
    if rel:
        min = average_entropy.min()
        max = average_entropy.max()
        average_entropy = (average_entropy - min) / (max - min)
    action = action.cpu().numpy()
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(action[0, :, 0], average_entropy, label=label)
    ax.set_xlabel("Action $u$")
    str = "Avg. \nentropy"
    if rel:
        str += "\n (norm.)"
    ax.set_ylabel(str)


def compute_average_entropy(
    context_encoder, initial_state, action, next_state, assignments, batchsize
):
    n_samples, n_actions = initial_state.shape[:2]
    context_encoding = context_encoder(
        initial_state.reshape(-1, initial_state.shape[-1]),
        action.reshape(-1, action.shape[-1]),
        next_state.reshape(-1, next_state.shape[-1]),
        assignments,
        batchsize,
    )
    entropy = context_encoding.entropy().sum(dim=-1).view(n_samples, n_actions)
    average_entropy = entropy.mean(dim=0)
    return average_entropy


def generate_context_set(env, initial_state, transition_model, context_encoder, device):
    n_actions = 500
    n_samples = 20

    assert len(env.action_space.shape) == 1
    action_range = (env.action_space.low[0], env.action_space.high[0])
    action = (
        torch.linspace(*action_range, steps=n_actions).unsqueeze(-1).float().to(device)
    )
    action = action[None, ...].expand(n_samples, *action.shape)
    initial_state = initial_state[None, None, ...].expand(
        n_samples, n_actions, *initial_state.shape
    )
    # Sample context variables for empty sets
    context = context_encoder.empty_set_context().mean[None, :]
    context = context[:, None, ...].expand(n_samples, n_actions, *context.shape[1:])
    next_state = transition_model(
        initial_state.contiguous(),
        action.contiguous(),
        context.contiguous(),
    ).mean

    batchsize = n_samples * n_actions
    assignments = torch.arange(batchsize, out=torch.LongTensor()).to(
        initial_state.device
    )
    return initial_state, action, next_state, assignments, batchsize


def initialize_env(env_name, device):
    env = make_env(env_name)
    env.initialize_context(111213)
    env.seed(111213)
    obs = env.reset(init_mode="calibration")
    initial_state = torch.from_numpy(obs).float().to(device)
    return env, initial_state


def average_entropy_plot_from_run(run_id, checkpoint_step, ax, label="", rel=False):
    device = "cuda"
    env_name, transition_model, context_encoder, log_likelihood_model = load_model(
        run_id, checkpoint_step, device
    )
    env, initial_state = initialize_env(env_name, device)
    with torch.no_grad():
        average_entropy_plot(
            env,
            initial_state,
            transition_model,
            context_encoder,
            device,
            ax,
            label,
            rel,
        )
