"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Independent, Normal

from context_exploration.utils import MLP


def get_transition_model(
    class_name, state_dim, action_dim, context_dim, embedding_dim, kwargs
):
    return globals()[class_name](
        state_dim=state_dim,
        action_dim=action_dim,
        context_dim=context_dim,
        embedding_dim=embedding_dim,
        **kwargs
    )


class GRUCellAnyBatch(nn.GRUCell):
    def forward(self, input, hx):
        if not input.shape[:-1] == hx.shape[:-1]:
            raise ValueError("Batch size of input and hx must match")
        bs = input.shape[:-1]
        input = input.view(-1, input.shape[-1])
        hx = hx.view(-1, hx.shape[-1])
        next_hx = super().forward(input, hx)
        next_hx = next_hx.view(*bs, next_hx.shape[-1])
        return next_hx


class TransitionModelBase(nn.Module):
    def __init__(self, state_dim, action_dim, context_dim, embedding_dim):
        super(TransitionModelBase, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
        self.device = None

    def to(self, device):
        self.device = device
        return super().to(device)


class GruTransitionModel(TransitionModelBase):
    def __init__(
        self,
        state_dim,
        action_dim,
        context_dim,
        embedding_dim,
        big_nets,
        hidden_dependent_noise=False,
    ):
        super(GruTransitionModel, self).__init__(
            state_dim, action_dim, context_dim, embedding_dim
        )

        self.state_embed_net = MLP(
            input_dim=state_dim,
            output_dim=embedding_dim,
            hidden_dims=[200, 200] if big_nets else [200],
            hidden_nonlinearities="ReLU",
            output_nonlinearity="Tanh",
        )

        self.action_embed_net = MLP(
            input_dim=action_dim,
            output_dim=embedding_dim,
            hidden_dims=[200, 200] if big_nets else [200],
            hidden_nonlinearities="ReLU",
            output_nonlinearity="ReLU",
        )

        context_embed_net_layers = [
            nn.Linear(context_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200) if big_nets else None,
            nn.ReLU() if big_nets else None,
            nn.Linear(200, embedding_dim),
            nn.ReLU(),
        ]
        self.context_embed_net = nn.Sequential(
            *[l for l in context_embed_net_layers if l]
        )

        decode_net_layers = [
            nn.Linear(embedding_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200) if big_nets else None,
            nn.ReLU() if big_nets else None,
            nn.Linear(200, state_dim),
        ]
        self.decode_net = nn.Sequential(*[l for l in decode_net_layers if l])

        # input: [embed(action), embed(context)]
        # initial hidden state: embed(state)
        self.gru_cell = GRUCellAnyBatch(
            input_size=2 * embedding_dim, hidden_size=embedding_dim
        )

        self.hidden_dependent_noise = hidden_dependent_noise
        if hidden_dependent_noise is True or hidden_dependent_noise == "mlp":
            self.hidden_cov_net = MLP(
                input_dim=embedding_dim,
                output_dim=state_dim,
                hidden_dims=[200],
                hidden_nonlinearities="ReLU",
                output_nonlinearity=None,
            )
        elif hidden_dependent_noise == "timedependent":
            # max prediction horizon: 100
            self.constant_cov_raw = nn.Parameter(0 * torch.ones(100, state_dim))
        elif hidden_dependent_noise is False or hidden_dependent_noise == "constant":
            self.constant_cov_raw = nn.Parameter(0 * torch.ones(state_dim))
        else:
            raise ValueError("Invalid hidden_dependent_noise")

        self.diag_cov_activation = lambda x: F.softplus(x) + 0.01 ** 2

    def forward_transition(self, x, u, ctx_mean):
        """
        Compute x' = f(x, u, ctx)

        Parameters
        ----------
        x : torch.Tensor, shape [<bs> x state_dim]
        u : torch.Tensor, shape [<bs> x action_dim]
        ctx : torch.Tensor, shape [<bs> x context_dim]

        Returns
        -------
        next_state : torch.Tensor, shape [<bs> x state_dim]
        """
        bs = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        u = u.view(-1, u.shape[-1])
        ctx_mean = ctx_mean.view(-1, ctx_mean.shape[-1])

        state_embed = self.state_embed_net(x)
        action_embed = self.action_embed_net(u)
        ctx_embed = self.context_embed_net(ctx_mean)

        next_hidden = self.gru_cell(
            input=torch.cat((ctx_embed, action_embed), dim=-1), hidden=state_embed
        )

        next_state = self.decode_net(next_hidden).view(*bs, self.state_dim)

        return next_state

    def forward_transition_multi_step(self, x, u, ctx_mean, return_hidden=False):
        """
        Compute x' = f(x, u1, ..., uN, ctx)

        Parameters
        ----------
        x : torch.Tensor, shape [<bs> x state_dim]
        u : torch.Tensor, shape [T x <bs> x action_dim]
        ctx_mean : torch.Tensor, shape [<bs> x context_dim]
        return_hidden : bool

        Returns
        -------
        next_state : torch.Tensor, shape [T x <bs> x state_dim]
        hidden_states : torch.Tensor, optional, shape [T x <bs> x state_dim]
        """
        assert x.shape[:-1] == u.shape[1:-1] == ctx_mean.shape[:-1]

        state_embed = self.state_embed_net(x)
        action_embed_all = self.action_embed_net(u)

        ctx_embed = self.context_embed_net(ctx_mean)

        hidden_state = state_embed
        hidden_state_list = [hidden_state]
        for T in range(u.shape[0]):
            action_embed = action_embed_all[T]
            hidden_state = self.gru_cell(
                torch.cat((ctx_embed, action_embed), dim=-1),
                hidden_state,
            )
            hidden_state_list.append(hidden_state)

        state_decoding_list = []
        for idx, hidden_state in enumerate(hidden_state_list):
            state_decoding = self.decode_net(hidden_state)
            state_decoding_list.append(state_decoding)

        returns = [torch.stack(state_decoding_list)]
        if return_hidden:
            returns.append(torch.stack(hidden_state_list))
        return tuple(returns) if len(returns) > 1 else returns[0]

    def forward_multi_step(
        self, x, u, ctx_mean, return_mean_only=False, probe_dict=None
    ):
        # skip uncertainty propagation
        state_mean, hidden_states = self.forward_transition_multi_step(
            x, u, ctx_mean, return_hidden=True
        )
        if return_mean_only:
            return state_mean

        expand = False
        if self.hidden_dependent_noise is True or self.hidden_dependent_noise == "mlp":
            covariance = self.diag_cov_activation(self.hidden_cov_net(hidden_states))
        elif self.hidden_dependent_noise == "timedependent":
            horizon = state_mean.shape[0]
            covariance = self.diag_cov_activation(self.constant_cov_raw)[:horizon, :]
            expand = True
        elif (
            self.hidden_dependent_noise is False
            or self.hidden_dependent_noise == "constant"
        ):
            # first dimension is for time
            covariance = self.diag_cov_activation(self.constant_cov_raw)[None, :]
            expand = True
        else:
            raise ValueError("Invalid hidden_dependent_noise")

        if expand:
            sliceop = (slice(None),) + (None,) * (state_mean.dim() - 2) + (slice(None),)
            covariance = covariance[sliceop].expand(*state_mean.shape)

        distribution = Independent(
            Normal(state_mean, torch.sqrt(covariance)), reinterpreted_batch_ndims=1
        )
        if probe_dict is not None:
            probe_dict["transition_model/ctx_mean_var"] = ctx_mean.var().item()
            probe_dict["transition_model/covariance"] = covariance.mean().item()
        return distribution

    def forward(self, x, u, ctx_mean, probe_dict=None):
        multi_step_result = self.forward_multi_step(
            x, u.unsqueeze(0), ctx_mean, probe_dict=probe_dict
        )
        predicted_next_states = Independent(
            Normal(multi_step_result.mean[1], multi_step_result.stddev[1]),
            reinterpreted_batch_ndims=1,
        )
        return predicted_next_states

    def sequence_logll(self, x, u, ctx_mean, probe_dict=None):
        """
        Compute sequence log-likelihood. Does not include reconstruction loss!

        Parameters
        ----------
        x : torch.Tensor, shape [T x <bs> x state_dim]
        u : torch.Tensor, shape [T x <bs> x action_dim]
        ctx_mean : torch.Tensor, shape [<bs> x context_dim]

        Returns
        -------
        sequence_logll : torch.Tensor, shape [<bs>]
        """
        prediction_distribution = self.forward_multi_step(
            x[0], u, ctx_mean, probe_dict=probe_dict
        )
        # strip off first state (does not include reconstruction loss)
        # strip off second state (does not include single-step loss)
        # strip off last prediction for which we do not have a ground-truth state
        pred_mean = prediction_distribution.mean[2:-1]
        pred_std = prediction_distribution.stddev[2:-1]
        prediction_distribution = Independent(
            Normal(pred_mean, pred_std),
            reinterpreted_batch_ndims=1,
        )
        sequence_logll = prediction_distribution.log_prob(x[2:]).sum(dim=0)
        return sequence_logll

    def single_step_logll(self, x, u, x_next, ctx_mean, probe_dict=None):
        """
        Compute reconstruction and single-step log-likelihood

        Parameters
        ----------
        x : torch.Tensor, shape [<bs> x state_dim]
        u : torch.Tensor, shape [<bs> x action_dim]
        x_next : torch.Tensor, shape [<bs> x action_dim]
        ctx_mean : torch.Tensor, shape [<bs> x context_dim]

        Returns
        -------
        reconstruction_logll: torch.Tensor, shape [<bs>]
        single_step_logll : torch.Tensor, shape [<bs>]
        """
        # introduce a fake time dimension
        u = u.unsqueeze(0)
        prediction_distribution = self.forward_multi_step(
            x, u, ctx_mean.contiguous(), probe_dict=probe_dict
        )
        # 'current state' is at index 0
        reconstructed_states = Independent(
            Normal(prediction_distribution.mean[0], prediction_distribution.stddev[0]),
            reinterpreted_batch_ndims=1,
        )
        # 'next_state' is at index 1
        predicted_next_states = Independent(
            Normal(prediction_distribution.mean[1], prediction_distribution.stddev[1]),
            reinterpreted_batch_ndims=1,
        )
        reconstruction_logll = reconstructed_states.log_prob(x)
        single_step_logll = predicted_next_states.log_prob(x_next)
        return reconstruction_logll, single_step_logll
