"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
from copy import deepcopy
import os
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.distributions import Normal, kl_divergence
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")

from context_exploration.data.envs import make_env
from context_exploration.data.gym_data_sampler import GymDataSampler
from context_exploration.evaluation.evaluation_helpers import (
    is_context_set_informative,
    plot_context_prediction,
    plot_rollouts,
)
from context_exploration.model.context_encoder import get_context_encoder
from context_exploration.model.transition_model import get_transition_model

ex = Experiment("train_model")
run_basedir = (
    Path(__file__).resolve().parent.parent.joinpath("experiments", "train_model")
)
print(f"Run basedir is at {run_basedir}")

os.makedirs(run_basedir, exist_ok=True)

ex.observers.append(FileStorageObserver(run_basedir))


@ex.config
def ex_cfg():
    seed = 42
    embedding_dim = 200
    context_encoder_classname = "MLPContextEncoder"
    context_encoder_kwargs = {
        "positive_weights": "relu",
        "aggregation_type": "max",
        "latent_dim": 32,
        "rescale_raw_stddev": True,
        "clamp_softplus_weights": True,
        "mean_feature_bidding": "none",
    }
    transition_model_classname = "GruTransitionModel"
    transition_model_kwargs = {
        "big_nets": False,
        "hidden_dependent_noise": True,
    }
    obs_n_transitions = 50
    grad_clip_norm = 1000
    learning_rate = 1e-3
    kl_empty_factor = 1.0
    kl_all_factor = 0
    sequence_logll_weight = 1
    kl_np_weight = 1
    inference_approx = "np"
    debug = False


@ex.named_config
def action_squash_cfg():
    env_name = "action_squash_1"
    n_rollouts_train = 5_000
    n_rollouts_val = 1_000
    n_training_steps = 50_000
    batchsize = 64
    max_ctx_card = 10
    context_dim = 1
    transition_model_kwargs = {"hidden_dependent_noise": False}
    ctx_from_train_prob_schedule = "constant"


@ex.named_config
def pendulum_bd_cfg():
    env_name = "pendulum_quadrantactionfactorar2bd"
    n_rollouts_train = 100_000
    n_rollouts_val = 10_000
    n_training_steps = 100_000
    batchsize = 512
    max_ctx_card = 50
    context_dim = 16
    context_encoder_kwargs = {"latent_dim": 128}
    ctx_from_train_prob_schedule = "linear"


@ex.named_config
def mountaincar_cfg():
    env_name = "mountaincar"
    n_rollouts_train = 50_000
    n_rollouts_val = 10_000
    n_training_steps = 100_000
    batchsize = 512
    max_ctx_card = 50
    context_dim = 16
    context_encoder_kwargs = {"latent_dim": 128}
    ctx_from_train_prob_schedule = "linear"


@ex.named_config
def no_positive_weights():
    context_encoder_kwargs = {"positive_weights": False, "rescale_raw_stddev": False}


def kl_standard_normal(p):
    # kl_divergence(p, N(0, 1))
    variance = p.scale.pow(2)
    return 0.5 * (variance + p.loc.pow(2) - 1 - variance.log())


def zip_source_dir(run_directory):
    source_dir = Path(__file__).resolve().parent.parent
    target_file = Path(run_directory).joinpath("source.tgz")
    os.system(f"tar cfz {target_file} {source_dir}")


def compute_loss(
    *,
    batch,
    batchsize,
    env,
    context_encoder,
    transition_model,
    device,
    inference_approx,
    kl_empty_factor,
    kl_all_factor,
    kl_np_weight,
    sequence_logll_weight,
):
    context_distribution = context_encoder(
        batch["ctx_x"],
        batch["ctx_u"],
        batch["ctx_x_next"],
        batch["ctx_assignments"],
        batchsize,
    )

    if inference_approx == "np":
        T, B, state_dim = batch["obs_x"].shape
        action_dim = batch["obs_u"].shape[-1]
        obs_x = batch["obs_x"][:-1].reshape(-1, state_dim)
        obs_u = batch["obs_u"][:-1].reshape(-1, action_dim)
        obs_x_next = batch["obs_x"][1:].reshape(-1, state_dim)
        obs_assignments = (
            torch.arange(batchsize).to(device)[None, :].expand(T - 1, B).reshape(-1)
        )
        ctx_obs_distribution = context_encoder(
            torch.cat((batch["ctx_x"], obs_x)),
            torch.cat((batch["ctx_u"], obs_u)),
            torch.cat((batch["ctx_x_next"], obs_x_next)),
            torch.cat((batch["ctx_assignments"], obs_assignments)),
            batchsize,
        )
        kld_np = kl_divergence(ctx_obs_distribution, context_distribution)
        kld_np = torch.clamp(kld_np, min=0.1)
        kld_np = kld_np.sum(dim=-1)
    else:
        raise ValueError(f"Unknown inference_approx {inference_approx}")

    empty_distribution = context_encoder(
        torch.zeros(0, env.state_dim).to(device),
        torch.zeros(0, env.action_dim).to(device),
        torch.zeros(0, env.state_dim).to(device),
        torch.zeros(0).long().to(device),
        batchsize=1,
    )

    assert kl_empty_factor >= 0
    if kl_empty_factor > 0:
        kld_empty = kl_standard_normal(empty_distribution).sum(dim=-1)
    else:
        kld_empty = torch.zeros(batchsize).to(device)

    if kl_all_factor > 0:
        kld_all = kl_standard_normal(context_distribution).sum(dim=-1)
    else:
        kld_all = torch.zeros(batchsize).to(device)

    sequence_logll_probes = {}
    if inference_approx == "np":
        sequence_logll = transition_model.sequence_logll(
            batch["obs_x"],
            batch["obs_u"],
            ctx_obs_distribution.rsample(),
            probe_dict=sequence_logll_probes,
        )
    else:
        raise ValueError(f"Unknown inference_approx {inference_approx}")

    T = len(batch["obs_x"])
    shape = context_distribution.mean.shape

    single_step_logll_probes = {}
    if inference_approx == "np":
        ctx_obs_distribution_expanded = Normal(
            ctx_obs_distribution.loc.expand(T - 1, *shape),
            ctx_obs_distribution.scale.expand(T - 1, *shape),
        )
        (reconstruction_logll, single_step_logll,) = transition_model.single_step_logll(
            batch["obs_x"][:-1],
            batch["obs_u"][:-1],
            batch["obs_x"][1:],
            ctx_obs_distribution_expanded.rsample(),
            probe_dict=single_step_logll_probes,
        )
        reconstruction_logll = reconstruction_logll.mean(dim=0)
        single_step_logll = single_step_logll.mean(dim=0)
    else:
        raise ValueError(f"Unknown inference_approx {inference_approx}")

    kld_empty_weighted = kld_empty * kl_empty_factor
    kld_all_weighted = kld_all * kl_all_factor
    kld_np_weighted = kld_np * kl_np_weight

    sequence_logll_weighted = sequence_logll * sequence_logll_weight

    observation_logll = (
        reconstruction_logll + single_step_logll + sequence_logll_weighted
    )

    loss_scalar = (
        -observation_logll.mean(dim=0)
        + kld_empty_weighted.mean(dim=0)
        + kld_all_weighted.mean(dim=0)
        + kld_np_weighted.mean(dim=0)
    )

    loss_dict = {
        "reconstruction_logll": reconstruction_logll.mean(dim=0),
        "single_step_logll": single_step_logll.mean(dim=0),
        "sequence_logll": sequence_logll.mean(dim=0),
        "sequence_logll_w": sequence_logll_weighted.mean(dim=0),
        "kld_empty": kld_empty.mean(dim=0),
        "kld_empty_w": kld_empty_weighted.mean(dim=0),
        "kld_all": kld_all.mean(dim=0),
        "kld_all_w": kld_all_weighted.mean(dim=0),
        "kld_np": kld_np.mean(dim=0),
        "kld_np_w": kld_np_weighted.mean(dim=0),
        "loss": loss_scalar,
    }

    probes = dict(
        **{f"single_step_logll/{k}": v for k, v in single_step_logll_probes.items()},
        **{f"sequence_logll_probes/{k}": v for k, v in sequence_logll_probes.items()},
    )

    intermediates = {"context_distribution": context_distribution}

    return loss_dict, probes, intermediates


def sample_validation_data(data_sampler, device):
    # Randomly sample 5 batches for validation
    validation_data = data_sampler.sample_validation_data(n_batches=5, device=device)
    return validation_data


def compute_validation_loss(validation_data, loss_kwargs):
    losses = defaultdict(list)
    for batch in validation_data:
        loss_dict, _, _ = compute_loss(batch=batch, **loss_kwargs)
        for k, v in loss_dict.items():
            losses[k].append(v.item())
    for k, v in losses.items():
        losses[k] = np.mean(v)
    return losses


def get_checkpoint_data(
    context_encoder, transition_model, validation_loss, step, is_best
):
    return {
        "transition_model": deepcopy(transition_model.state_dict()),
        "context_encoder": deepcopy(context_encoder.state_dict()),
        "validation_loss": validation_loss,
        "step": step,
        "is_best": is_best,
    }


def linear_slope(step, step_start, step_end, value_start, value_end):
    relative_step = np.clip((step - step_start) / (step_end - step_start), 0, 1)
    value = value_start + relative_step * (value_end - value_start)
    return value


@ex.automain
def main(
    env_name,
    embedding_dim,
    context_dim,
    context_encoder_classname,
    context_encoder_kwargs,
    transition_model_classname,
    transition_model_kwargs,
    n_rollouts_train,
    n_rollouts_val,
    obs_n_transitions,
    max_ctx_card,
    grad_clip_norm,
    learning_rate,
    batchsize,
    n_training_steps,
    debug,
    kl_empty_factor,
    kl_all_factor,
    sequence_logll_weight,
    inference_approx,
    kl_np_weight,
    ctx_from_train_prob_schedule,
    _run,
):
    device = "cuda"

    run_directory = os.path.join(run_basedir, _run._id)
    # zip_source_dir(run_directory)

    env = make_env(env_name)
    if env.context_dim != context_dim:
        warnings.warn("Non-matching context dim")

    if debug:
        n_rollouts_train = 1000
        n_rollouts_val = 1000

    ctx_card = list(range(0, max_ctx_card + 1))
    sampler_train = GymDataSampler(
        env_name,
        env.state_dim,
        env.action_dim,
        obs_n_transitions,
        ctx_card,
        batchsize,
        seed_min_incl=0,
        seed_max_incl=n_rollouts_train - 1,
        ctx_from_train_prob=0,
    )

    sampler_val = GymDataSampler(
        env_name,
        env.state_dim,
        env.action_dim,
        obs_n_transitions,
        ctx_card,
        batchsize,
        seed_min_incl=n_rollouts_train,
        seed_max_incl=n_rollouts_train + n_rollouts_val - 1,
        ctx_from_train_prob=0,
    )

    transition_model = get_transition_model(
        transition_model_classname,
        env.state_dim,
        env.action_dim,
        context_dim,
        embedding_dim,
        transition_model_kwargs,
    ).to(device)

    context_encoder = get_context_encoder(
        context_encoder_classname,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        context_dim=context_dim,
        kwargs=context_encoder_kwargs,
    ).to(device)

    dynamics_model_parameters = list(transition_model.parameters()) + list(
        context_encoder.parameters()
    )
    optim_trans = torch.optim.Adam(
        [
            {"params": transition_model.parameters(), "lr": learning_rate, "eps": 1e-4},
            {"params": context_encoder.parameters(), "lr": learning_rate, "eps": 1e-4},
        ]
    )

    writer = SummaryWriter(run_directory)

    validation_data = sample_validation_data(sampler_val, device)
    best_validation_loss = None
    best_checkpoint_data = None

    model_directory = os.path.join(run_directory, "models")
    os.makedirs(model_directory, exist_ok=True)

    def _log_scalar(_tag, _value, _step):
        writer.add_scalar(_tag, _value, _step)
        _run.log_scalar(_tag, _value, _step)

    loss_kwargs = {
        "env": env,
        "context_encoder": context_encoder,
        "transition_model": transition_model,
        "batchsize": batchsize,
        "device": device,
        "inference_approx": inference_approx,
        "kl_empty_factor": kl_empty_factor,
        "kl_all_factor": kl_all_factor,
        "kl_np_weight": kl_np_weight,
        "sequence_logll_weight": sequence_logll_weight,
    }

    for step in range(n_training_steps):
        if ctx_from_train_prob_schedule == "linear":
            ctx_from_train_prob = linear_slope(
                step, step_start=30_000, step_end=60_000, value_start=0.5, value_end=0
            )

        elif ctx_from_train_prob_schedule == "constant":
            ctx_from_train_prob = 0
        else:
            raise ValueError(
                f"Unknown 'ctx_from_train_prob_schedule': "
                f"{ctx_from_train_prob_schedule}"
            )

        sampler_train.ctx_from_train_prob = ctx_from_train_prob
        _log_scalar("slopes/ctx_from_train_prob", ctx_from_train_prob, step)

        batch = sampler_train.sample_batch(device=device)

        loss_dict, probes, intermediates = compute_loss(batch=batch, **loss_kwargs)

        if (step + 1) % 100 == 0:
            for k, v in loss_dict.items():
                _log_scalar(f"loss/{k}", v.item(), step)
            for k, v in probes.items():
                _log_scalar(k, v, step)

        optim_trans.zero_grad()
        loss = loss_dict["loss"]
        loss.backward()
        if grad_clip_norm is not None:
            clip_grad_norm_(dynamics_model_parameters, grad_clip_norm, norm_type=2)
        optim_trans.step()

        if step == 0:
            print("Starting training, printing info every 100th step")
        if (step + 1) % 100 == 0:
            print(step + 1, loss_dict)

        # Plot
        if (step == 0) or (step + 1) % 1000 == 0:
            is_informative = is_context_set_informative(
                env,
                batch["ctx_x"].cpu().numpy(),
                batch["ctx_u"].cpu().numpy(),
                batch["ctx_x_next"].cpu().numpy(),
                batch["ctx_assignments"].cpu().numpy(),
                batchsize,
            )
            context_sse = None
            plot_context_prediction(
                intermediates["context_distribution"],
                is_informative,
                batch["ctx_size"].cpu().numpy(),
                context_sse,
                step + 1,
                run_directory,
                writer,
            )
            plot_rollouts(
                env_name,
                transition_model,
                context_encoder,
                step + 1,
                run_directory,
                writer,
            )

        # Compute validation loss
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                validation_loss_dict = compute_validation_loss(
                    validation_data, loss_kwargs
                )
            for k, v in validation_loss_dict.items():
                _log_scalar(f"validation/{k}", v, step)
            validation_loss = validation_loss_dict["loss"]
            if best_validation_loss is None or (validation_loss < best_validation_loss):
                best_validation_loss = validation_loss
                for k, v in validation_loss_dict.items():
                    _log_scalar(f"best_validation/{k}", v, step)
                print(f"New best validation result: {validation_loss}")
                best_checkpoint_data = get_checkpoint_data(
                    context_encoder,
                    transition_model,
                    validation_loss,
                    step,
                    is_best=True,
                )

        # Checkpoint current and best model
        if (step + 1) % 5000 == 0:
            # Store current model
            checkpoint_file = os.path.join(
                model_directory, f"checkpoint_step_{step+1}.pkl"
            )
            checkpoint_data = get_checkpoint_data(
                context_encoder, transition_model, validation_loss, step, is_best=False
            )
            torch.save(checkpoint_data, checkpoint_file)
            # Store model which is best up to the current step
            checkpoint_file_best = os.path.join(
                model_directory, f"checkpoint_step_{step + 1}_best.pkl"
            )
            torch.save(best_checkpoint_data, checkpoint_file_best)
