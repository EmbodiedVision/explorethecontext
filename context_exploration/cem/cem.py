"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import warnings

import numpy as np
import torch
from numpy import newaxis, sqrt
from numpy import sum as np_sum
from numpy.fft import irfft, rfftfreq
from numpy.random import normal
from torch import nn
from torch.distributions.distribution import Distribution
from tqdm import tqdm


class Belief(Distribution):
    @property
    def device(self):
        return self.mean.device


class AbstractTransitionModel(nn.Module):
    def __init__(self):
        super(AbstractTransitionModel, self).__init__()

    @property
    def action_size(self):
        raise NotImplementedError

    def multi_step(self, initial_state_belief: Belief, actions: torch.Tensor) -> Belief:
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
        state_beliefs: Belief with batch_shape [(T+1) x <bs>]
            State beliefs including initial_state_belief
        """
        raise NotImplementedError

    def _check_dimensions(self, initial_state_belief: Belief, actions: torch.Tensor):
        if not actions.shape[-1] == self.action_size:
            raise ValueError("Invalid action size")
        if not initial_state_belief.batch_shape == actions.shape[1:-1]:
            raise ValueError("Invalid batch shape")


class AbstractReturnModel(nn.Module):
    def __init__(self):
        super(AbstractReturnModel, self).__init__()

    def forward(self, state_belief: Belief, action: torch.Tensor) -> torch.Tensor:
        """
        Return for applying action sequence leading to state beliefs

        Parameters
        ----------
        state_belief: [T+1 x <bs> x belief_size]
            States including initial state
        action: [T x <bs> x action_size]
            Applied actions (first action is applied to first state belief)

        Returns
        -------
        returns: torch.Tensor [<bs>, 1]
        """
        raise NotImplementedError

    def _check_dimensions(self, state_belief: Belief, action: torch.Tensor):
        if not (
            state_belief.batch_shape[0] == action.shape[0] + 1
            and state_belief.batch_shape[1:] == action.shape[1:-1]
        ):
            raise ValueError("Invalid batch shape")


class CEM(nn.Module):
    def __init__(
        self,
        transition_model,
        return_model,
        planning_horizon,
        action_space,
        optimisation_iters=10,
        candidates=1000,
        top_candidates=100,
        clip_actions=True,
        return_all_actions=False,
        return_mean=True,
        verbose=False,
    ):
        """
        MPCPlannerCem

        Parameters
        ----------
        transition_model: TransitionModel
            Transition model
        return_model: ReturnModel
            Return model
        planning_horizon: int
            Planning horizon
        action_space: Box:
            Action space
        optimisation_iters: int
            Number of CEM iterations
        candidates: int
            Number of candidates per iteration
        top_candidates: int
            Number of best candidates to refit belief per iteration
        clip_actions: bool
            Clip actions to action_space range
        return_all_actions: bool, default False
            Return all actions instead of the first one to apply
        return_mean: bool, default True
            If True, return mean of best actions; if False,
            return best action sequence
        verbose: bool, default False
            Be verbose
        """
        super().__init__()
        self.transition_model, self.return_model = transition_model, return_model
        self.planning_horizon = planning_horizon
        self.action_space = action_space
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates
        self.clip_actions = clip_actions
        self.return_all_actions = return_all_actions
        self.return_mean = return_mean
        self.verbose = verbose

    def forward(self, initial_state_belief):
        """
        Compute optimal action for current state

        Parameters
        ----------
        initial_state_belief: Belief with batch_shape <bs>
            Distribution of initial state
        """
        action_size = self.transition_model.action_size
        device = initial_state_belief.device
        expanded_state_belief = initial_state_belief.expand(
            initial_state_belief.batch_shape + torch.Size([self.candidates])
        )
        # expanded_state_belief: <batch> x n_candidates x <variable_dim>
        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        action_range = [
            torch.Tensor(self.action_space.low).to(device),
            torch.Tensor(self.action_space.high).to(device),
        ]
        assert all(-action_range[0] == action_range[1])
        action_belief_shape = [
            self.planning_horizon,
            *initial_state_belief.batch_shape,
            1,
            action_size,
        ]
        action_sample_shape = [
            self.planning_horizon,
            *initial_state_belief.batch_shape,
            self.candidates,
            action_size,
        ]
        action_mean, action_std_dev = (
            torch.zeros(*action_belief_shape, device=device),
            torch.ones(*action_belief_shape, device=device) * action_range[1],
        )

        iterable = list(range(self.optimisation_iters))
        if self.verbose:
            iterable = tqdm(iterable)
        for _ in iterable:
            # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
            # Sample actions [T x <bs> x n_candidates x action_dim]
            random_samples = torch.randn(
                *action_sample_shape, device=action_mean.device
            )
            actions = action_mean + action_std_dev * random_samples
            if self.clip_actions:
                actions = torch.where(
                    actions < action_range[0], action_range[0], actions
                )
                actions = torch.where(
                    actions > action_range[1], action_range[1], actions
                )
            # actions: [T x <bs> x n_candidates x action_dim]
            # Sample next states
            # Plan in latent space
            next_state_belief = self.transition_model.multi_step(
                expanded_state_belief, actions
            )

            # next_state_belief: batch_shape [(T+1) x <bs> x n_candidates]
            # Calculate expected returns (technically sum of rewards over planning horizon)
            returns = self.return_model.forward(next_state_belief, actions).unsqueeze(0)

            # returns: [1 x <bs> x n_candidates x 1]
            # Re-fit action belief to the K best action sequences
            # If the best action sequence should be return, sort
            _, topk = returns.topk(
                self.top_candidates, dim=-2, largest=True, sorted=not self.return_mean
            )
            topk = topk.expand(*actions.shape[:-2], topk.shape[-2], actions.shape[-1])
            best_actions = torch.gather(actions, dim=-2, index=topk)
            # best_actions = [T x <bs> x top_candidates x action_size]

            # Update belief with new means and standard deviations
            action_mean = best_actions.mean(
                dim=-2, keepdim=True
            )  # Mean of all candidates
            # action_mean = T x <bs> x 1 x action_size
            action_std_dev = best_actions.std(dim=-2, unbiased=False, keepdim=True)

        planner_info = {}
        if self.return_mean:
            action_sequence = action_mean[..., 0, :]
        else:
            action_sequence = best_actions[..., 0, :]

        if self.return_all_actions:
            return action_sequence, planner_info
        else:
            return action_sequence[0], planner_info
