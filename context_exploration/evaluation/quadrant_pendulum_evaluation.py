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

from context_exploration.data.envs import make_env
from context_exploration.model.context_encoder import ContextSet
from context_exploration.model.loader import load_model


def angular_plot(angle, angular_velocity, ax):
    """

    Parameters
    ----------
    angle : np.ndarray, [n_contexts x <bs>]
    angular_velocity : np.ndarray, [n_contexts x <bs>]
    """
    n_contexts = angle.shape[0]
    context_idx = np.broadcast_to(
        np.arange(n_contexts)[(slice(None),) + (None,) * (angle.ndim - 1)], angle.shape
    ).flatten()
    x = np.cos(angle + np.pi / 2).flatten()
    y = np.sin(angle + np.pi / 2).flatten()
    r = 1 + 0.2 * angular_velocity.flatten()

    c = plt.cm.RdYlBu(context_idx / n_contexts)
    ax.scatter(x * r, y * r, color=c)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])


def quadrant_pendulum_entropy_plot(
    env,
    context_encoder,
    ax_0=None,
    ax_1=None,
    ylim=None,
    scatter_rad=10,
    scatter_alpha=0.5,
    scatter_marker="o",
    text_y=-6.5,
    fontsize=8,
):
    n_contexts = 5
    per_quadrant_samples = 15

    if ax_0 is None:
        fig, ax_arr = plt.subplots(nrows=1, ncols=2)
        ax_0 = ax_arr[0]
        ax_1 = ax_arr[1]

    sample_gen = np.random.RandomState(1)

    angle_samples = []
    for quadrant in [0, 1, 2, 3]:
        # First quadrant is "1"
        quadrant = (quadrant + 1) % 4
        samples = sample_gen.uniform(
            -np.pi + quadrant * np.pi / 2 + np.pi / 16,
            -np.pi + (quadrant + 1) * np.pi / 2 - np.pi / 16,
            (n_contexts, per_quadrant_samples),
        )
        angle_samples.append(samples)
    angle_samples = np.stack(angle_samples, axis=1)
    # angle_samples: n_contexts x quadrant x per_quadrant_samples
    # sample uniformly from -4..-2, 2..4
    velocity_samples = sample_gen.uniform(2, 4, angle_samples.shape)
    velocity_samples *= sample_gen.choice(np.array([-1, 1]), angle_samples.shape)
    actions = sample_gen.uniform(-1, 1, angle_samples.shape)[..., None]
    if ax_1:
        angular_plot(angle_samples, velocity_samples, ax_1)
    initial_states = np.stack((angle_samples, velocity_samples), axis=-1)

    context_seed_gen = np.random.RandomState(543)
    entropies = np.zeros((n_contexts, 4 * per_quadrant_samples + 1))
    for context_idx in range(n_contexts):
        context_seed = context_seed_gen.randint(0, int(1e8))
        transitions = generate_transitions(env, context_seed, initial_states, actions)
        # transitions: (x: quadrant x per_quadrant_samples x state_dim, u, x_next)
        transitions = [t.reshape(-1, t.shape[-1]) for t in transitions]
        # transitions: (x: quadrant * per_quadrant_samples x state_dim, u, x_next)
        for set_size in range(4 * per_quadrant_samples + 1):
            limited_transitions = [t[:set_size] for t in transitions]
            context_set = ContextSet.from_array(*limited_transitions)
            encoding = context_encoder.forward_set(context_set)
            entropies[context_idx, set_size] = encoding.entropy().sum(dim=-1)

    for context_idx in range(n_contexts):
        ax_0.scatter(
            np.arange(1),
            entropies[context_idx][:1],
            s=20 * scatter_rad,
            color="r",
            marker=scatter_marker,
        )
        ax_0.scatter(
            np.arange(1, 4 * per_quadrant_samples + 1),
            entropies[context_idx][1:],
            s=scatter_rad,
            color="b",
            alpha=scatter_alpha,
            marker=scatter_marker,
        )

    if ylim:
        ax_0.set_ylim(*ylim)
    else:
        ax_0.set_ylim(-7, 12)

    ax_0.set_xlim(-2, per_quadrant_samples * 4)
    texts = ["Q1", "Q1+Q2", "Q1+Q2\n+Q3", "Q1+Q2\n+Q3+Q4"]
    for quadrant in range(4):
        text = texts[quadrant]
        ax_0.text(
            x=quadrant * per_quadrant_samples + 2, y=text_y, s=text, fontsize=fontsize
        )
        ax_0.axvline(
            x=0.5 + quadrant * per_quadrant_samples,
            ymin=0,
            ymax=1,
            color="k",
            alpha=0.5,
        )


def generate_transitions(env, context_seed, initial_states, actions):
    assert initial_states.shape[:-1] == actions.shape[:-1]
    env.initialize_context(context_seed)
    states_flat = initial_states.reshape(-1, initial_states.shape[-1])
    actions_flat = actions.reshape(-1, actions.shape[-1])
    obs = []
    obs_next = []

    base_env = env
    while not (base_env.unwrapped == base_env):
        base_env = base_env.unwrapped

    for state, action in zip(states_flat, actions_flat):
        env.reset()
        base_env.state = state
        obs.append(env.unwrapped.get_obs())
        assert np.allclose(np.arctan2(obs[-1][1], obs[-1][0]), state[0])
        assert np.allclose(obs[-1][2], state[1])
        obs_next.append(env.step(action)[0])
    obs = np.stack(obs).reshape(*initial_states.shape[:-1], obs[0].shape[-1])
    obs_next = np.stack(obs_next).reshape(
        *initial_states.shape[:-1], obs_next[0].shape[-1]
    )
    env.release_context()
    return obs, actions, obs_next


if __name__ == "__main__":
    run_id = "cr_s1_pendulum_bd_posweights_relu_npklw5"
    checkpoint_step = "100000_best"
    device = "cuda"
    env_name, transition_model, context_encoder, log_likelihood_model = load_model(
        run_id, checkpoint_step, device
    )
    env = make_env(env_name)
    quadrant_pendulum_entropy_plot(env, context_encoder, ylim=(5, 25), text_y=5.8)
    plt.show()
