"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import numpy as np
import torch
from tqdm import tqdm

from context_exploration.cem.cem import CEM
from context_exploration.evaluation.evaluation_helpers import generate_context_set
from context_exploration.model.context_encoder import ContextSet


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
    def __init__(
        self,
        transition_model,
        *,
        context_sample=None,
        context_distribution=None,
        n_transition_samples=None
    ):
        """
        Transition model for CEM

        Parameters
        ----------
        transition_model : TransitionModel
        context_sample : torch.Tensor, [n_context_samples x B x context_dim]
        context_distribution
        """
        assert bool(context_sample is None) ^ bool(context_distribution is None)
        self.transition_model = transition_model
        if context_distribution is not None:
            assert n_transition_samples is not None
        self.n_transition_samples = n_transition_samples
        self.context_sample = context_sample
        self.context_distribution = context_distribution
        self.action_size = self.transition_model.action_dim

    def multi_step(self, initial_cem_state, action_sequence):
        """
        Construct context sets from rollouts starting at the given initial state

        Parameters
        ----------
        initial_cem_state : InitialCemState, shape [B x n_candidates x state_dim]
        action_sequence : torch.Tensor, [T x B x n_candidates x action_dim]

        Returns
        -------
        x: torch.Tensor, [n_context_samples x B x n_candidates x T x state_dim]
        u: torch.Tensor, [n_context_samples x B x n_candidates x T x action_dim]
        x_next: torch.Tensor, [n_context_samples x B x n_candidates x T x state_dim]
        """
        if self.context_sample is not None:
            return self.multi_step_sample_context(initial_cem_state, action_sequence)
        elif self.context_distribution is not None:
            return self.multi_step_prop_context(initial_cem_state, action_sequence)
        else:
            raise ValueError

    def multi_step_sample_context(self, initial_cem_state, action_sequence):
        current_state = initial_cem_state.state
        n_context_samples, batchsize, context_dim = self.context_sample.shape
        n_candidates, state_dim = current_state.shape[1:]
        # initial state is identical for all context samples
        current_state = current_state.expand(n_context_samples, *current_state.shape)
        # current_state: [n_context_samples x B x n_candidates x state_dim]
        # context samples are identical for all action candidates
        context_sample = self.context_sample[:, :, None, :].expand(
            -1, -1, n_candidates, -1
        )
        # context_sample: [n_context_samples x B x context_dim x n_candidates]
        # actions are identical for all context samples
        action_sequence = action_sequence[:, None, :, :, :].expand(
            -1, n_context_samples, -1, -1, -1
        )
        # action_sequence: [T x n_context_samples x B x n_candidates x action_dim]
        assert (
            action_sequence.shape[1:-1]
            == current_state.shape[:-1]
            == context_sample.shape[:-1]
        )
        state_prediction = self.transition_model.forward_multi_step(
            current_state,
            action_sequence,
            context_sample.contiguous(),
            return_mean_only=True,
        )
        # state_prediction: [(T+1) x n_context_samples x B x n_candidates x state_dim]
        x = state_prediction[:-1]
        u = action_sequence
        x_next = state_prediction[1:]
        # x: [T x n_context_samples x B x n_candidates x state_dim]
        return x, u, x_next

    def multi_step_prop_context(self, initial_cem_state, action_sequence):
        current_state = initial_cem_state.state
        assert current_state.dim() == 3
        n_candidates = current_state.shape[1]
        context_distribution = self.context_distribution
        context_distribution = context_distribution.expand(
            (1, n_candidates, context_distribution.batch_shape[-1])
        )
        state_prediction = self.transition_model.forward_multi_step(
            current_state,
            action_sequence,
            context_distribution.mean,
            context_distribution.variance,
        )
        state_prediction = state_prediction.sample((self.n_transition_samples,))
        # states: n_transition_samples x (T+1) x B x n_candidates x state_dim
        state_prediction = state_prediction.permute(1, 0, 2, 3, 4)
        # states: (T+1) x n_transition_samples x B x n_candidates x state_dim
        x = state_prediction[:-1]
        # add n_transition_samples dimension to action_sequence
        u = action_sequence[:, None, :, :, :].expand(
            -1, self.n_transition_samples, -1, -1, -1
        )
        x_next = state_prediction[1:]
        # x: [T x n_transition_samples x B x n_candidates x state_dim]
        return x, u, x_next


class CemReturnModel:
    def __init__(
        self, prior_context_set, context_encoder, log_likelihood_model, criterion
    ):
        """
        Return model for CEM

        Parameters
        ----------
        log_likelihood_model
        """
        self.prior_context_set = prior_context_set.as_torch().to(context_encoder.device)
        self.context_encoder = context_encoder
        self.log_likelihood_model = log_likelihood_model
        self.criterion = criterion

    def forward(self, context_set, actions):
        """
        Compute 'return' (here: log-likelihood) for context sets

        Parameters
        ----------
        context_set: tuple of torch.Tensor,
            [n_actions x n_context_samples x B x n_candidates x state_dim]
        actions: unused

        Returns
        -------
        log_likelihood_avg: torch.Tensor, [B x n_candidates]
        """
        x, u, x_next = context_set
        # x: [T x n_context_samples x B x n_candidates x state_dim]

        if self.prior_context_set.is_empty:
            context_distribution = self.context_encoder.forward_tensor(x, u, x_next)
        else:
            context_distribution = self.context_encoder.forward_broadcast(
                x,
                u,
                x_next,
                self.prior_context_set.x[:, None, None, None, :],
                self.prior_context_set.u[:, None, None, None, :],
                self.prior_context_set.x_next[:, None, None, None, :],
            )

        assert self.criterion == "neg-entropy"
        objective = -context_distribution.entropy().sum(dim=-1, keepdim=True)
        # objective: [n_context_samples x B x n_candidates x 1]
        objective_avg = torch.mean(objective, dim=0)
        # objective_avg: [B, n_candidates, 1]
        return objective_avg


def optimize_action_sequence(
    initial_state,
    context_set,
    transition_model,
    context_encoder,
    log_likelihood_model,
    action_space,
    planning_horizon,
    n_context_samples,
    criterion,
    propagate_context,
    cem_kwargs=None,
):
    """
    Generate pseudo-trajectories from initial_state, provided
    actions and already observed context transitions. Then,
    build context sets from those trajectories and compute the
    marginal data log-likelihood. Average it over the context samples.

    Parameters
    ----------
    initial_state : torch.Tensor, [state_dim]
    context_set : ContextSet
        ContextSet of already observed transitions
    transition_model : TransitionModel
    context_encoder : ContextEncoder
    log_likelihood_model : LogLikelihoodModel
    action_space : gym.spaces.Space
    planning_horizon : int
    n_context_samples : int
    criterion : str, in ["neg-entropy", "log-likelihood"]
    propagate_context : bool
    cem_kwargs : dict, optional

    Returns
    -------

    """
    assert initial_state.dim() == 1
    # introduce batch dimension
    initial_state = initial_state.unsqueeze(0)
    # initial_state: [1 x state_dim]

    context_distribution = context_encoder.forward_set(context_set)
    context_distribution = context_distribution.expand((1, context_encoder.context_dim))
    if propagate_context:
        cem_transition_model = CemTransitionModel(
            transition_model,
            context_distribution=context_distribution,
            n_transition_samples=n_context_samples,
        )
    else:
        context_sample = context_distribution.sample((n_context_samples,))
        # context_sample: [n_context_samples x 1 x context_dim]
        cem_transition_model = CemTransitionModel(
            transition_model, context_sample=context_sample
        )

    cem_initial_state = InitialCemState(initial_state)

    cem_return_model = CemReturnModel(
        context_set, context_encoder, log_likelihood_model, criterion
    )

    cem_kwargs = {} if cem_kwargs is None else cem_kwargs
    cem = CEM(
        cem_transition_model,
        cem_return_model,
        planning_horizon=planning_horizon,
        action_space=action_space,
        return_all_actions=True,
        **cem_kwargs
    )

    with torch.no_grad():
        optimal_actions, _ = cem.forward(cem_initial_state)

    return optimal_actions


def optimize_action_sequence_openloop_multistart(
    env,
    initial_state_base_seed,
    n_env_inits,
    rollout_length,
    transition_model,
    context_encoder,
    log_likelihood_model,
    criterion,
    propagate_context,
    n_context_samples,
    cem_kwargs,
    verbose=False,
):
    assert n_env_inits > 0
    device = transition_model.device
    current_context_set = ContextSet.create_empty()
    initial_state_seed_gen = np.random.RandomState(initial_state_base_seed)
    for _ in range(n_env_inits):
        initial_state_seed = initial_state_seed_gen.randint(0, int(1e8))
        env.seed(initial_state_seed)
        obs = env.reset(init_mode="calibration")
        initial_state = torch.from_numpy(obs).float().to(device)
        optimal_actions = (
            optimize_action_sequence(
                initial_state,
                current_context_set,
                transition_model,
                context_encoder,
                log_likelihood_model,
                env.action_space,
                planning_horizon=rollout_length,
                n_context_samples=n_context_samples,
                criterion=criterion,
                propagate_context=propagate_context,
                cem_kwargs=cem_kwargs,
            )[
                :, 0, :
            ]  #  remove introduced batch dimension ([T x B x action_dim])
            .cpu()
            .numpy()
        )
        optimal_context_set = generate_context_set(
            env, env_seed=initial_state_seed, actions=optimal_actions
        )
        current_context_set += optimal_context_set

    optimal_context_latent = context_encoder.forward_set(current_context_set)
    return optimal_context_latent, None


def optimize_action_sequence_mpc_multistart(
    env,
    initial_state_base_seed,
    n_env_inits,
    rollout_length,
    transition_model,
    context_encoder,
    log_likelihood_model,
    planning_horizon,
    criterion,
    propagate_context,
    n_context_samples,
    cem_kwargs,
    verbose,
):
    assert n_env_inits > 0
    current_context_set = ContextSet.create_empty()
    initial_state_seed_gen = np.random.RandomState(initial_state_base_seed)
    rollout_list = []
    for _ in range(n_env_inits):
        initial_state_seed = initial_state_seed_gen.randint(0, int(1e8))
        optimal_context_latent, rollout = optimize_action_sequence_mpc(
            env,
            initial_state_seed,
            rollout_length,
            transition_model,
            context_encoder,
            log_likelihood_model,
            planning_horizon,
            criterion,
            propagate_context,
            n_context_samples,
            cem_kwargs,
            current_context_set,
            verbose,
        )
        rollout_list.append(rollout)
        current_context_set = current_context_set + rollout["additional_context_set"]

    return optimal_context_latent, rollout_list


def optimize_action_sequence_mpc(
    env,
    initial_state_seed,
    rollout_length,
    transition_model,
    context_encoder,
    log_likelihood_model,
    planning_horizon,
    criterion,
    propagate_context,
    n_context_samples,
    cem_kwargs,
    offset_context_set,
    verbose,
):
    env.seed(initial_state_seed)
    device = transition_model.device
    obs = env.reset(init_mode="calibration")
    current_state = torch.from_numpy(obs).float().to(device)
    observations = [obs]
    applied_actions = []
    current_context_distribution = context_encoder.forward_set(offset_context_set)
    entropy_history = []
    domain_history = [env.get_domain()]
    planned_actions = []
    current_context_set = offset_context_set
    for step in tqdm(range(rollout_length)):
        optimal_actions = (
            optimize_action_sequence(
                current_state,
                current_context_set,
                transition_model,
                context_encoder,
                log_likelihood_model,
                env.action_space,
                # We use a receding horizon towards the rollout length here.
                # The planning horizon should be upper bounded by 'planning_horizon'.
                # At the last step, the planning horizon should be 1.
                planning_horizon=min(planning_horizon, rollout_length - step),
                n_context_samples=n_context_samples,
                criterion=criterion,
                propagate_context=propagate_context,
                cem_kwargs=cem_kwargs,
            )[
                :, 0, :
            ]  # remove introduced batch dimension ([T x B x action_dim])
            .cpu()
            .numpy()
        )
        planned_actions.append(optimal_actions)
        first_action = optimal_actions[0]
        entropy = float(current_context_distribution.entropy().sum(dim=-1))
        entropy_history.append(entropy)
        new_obs, _, done, _ = env.step(first_action)
        assert not done
        if verbose:
            print(
                ", ".join(
                    ["{:10.4f}".format(float(a)) for a in optimal_actions[..., 0]]
                )
            )
            print(first_action, entropy)
            env.render(mode="human")
        applied_actions.append(first_action)
        observations.append(new_obs)
        domain_history.append(env.get_domain())
        current_state = torch.from_numpy(new_obs).float().to(device)
        additional_context_set = ContextSet.from_trajectory(
            np.stack(observations), np.stack(applied_actions)
        )
        current_context_set = offset_context_set + additional_context_set
        current_context_distribution = context_encoder.forward_set(current_context_set)

    domain_counts = [
        len(np.unique(domain_history[:k])) for k in range(len(domain_history))
    ]

    rollout = {
        "observation": np.stack(observations),
        "action": np.stack(applied_actions),
        "context_entropy": np.stack(entropy_history),
        "domain_history": np.stack(domain_history),
        "domain_counts": domain_counts,
        "planned_actions": planned_actions,
        "additional_context_set": additional_context_set,
    }

    return current_context_distribution, rollout
