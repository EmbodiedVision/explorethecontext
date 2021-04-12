"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import json
import os
import random
import string
from pathlib import Path

import torch

from context_exploration.data.envs import make_env
from context_exploration.model.context_encoder import (
    MLPContextEncoder,
    get_context_encoder,
)
from context_exploration.model.transition_model import get_transition_model


def get_run_directory(run_id):
    base_directory = Path(__file__).parent.parent.parent.joinpath(
        "experiments", "train_model"
    )
    run_directory = base_directory.joinpath(str(run_id))
    if not os.path.isdir(run_directory):
        # check if it occurs as a comment
        found = False
        for subdirectory in base_directory.iterdir():
            if not subdirectory.joinpath("run.json").is_file():
                continue
            with open(subdirectory.joinpath("run.json"), "r") as handle:
                run_data = json.load(handle)
            comment = run_data["meta"]["comment"]
            if comment == run_id:
                run_directory = subdirectory
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Run with id {run_id} cannot be located")
    return run_directory


def load_config(run_id):
    run_directory = get_run_directory(run_id)
    with open(os.path.join(run_directory, "config.json"), "r") as handle:
        config = json.load(handle)
    return config


def load_model(run_id, step, device):
    run_directory = get_run_directory(run_id)
    config = load_config(run_id)

    env_name = config["env_name"]
    env = make_env(env_name)

    model_file = os.path.join(run_directory, "models", f"checkpoint_step_{step}.pkl")

    transition_model = get_transition_model(
        config["transition_model_classname"],
        env.state_dim,
        env.action_dim,
        config["context_dim"],
        config["embedding_dim"],
        config["transition_model_kwargs"],
    )

    transition_model_state = torch.load(model_file)["transition_model"]
    transition_model.load_state_dict(transition_model_state)
    transition_model = transition_model.to(device)

    encoder_kwargs = config["context_encoder_kwargs"]
    context_encoder = get_context_encoder(
        config["context_encoder_classname"],
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        context_dim=config["context_dim"],
        kwargs=encoder_kwargs,
    )
    context_encoder_state = torch.load(model_file)["context_encoder"]
    context_encoder.load_state_dict(context_encoder_state)
    context_encoder = context_encoder.to(device)

    log_likelihood_model = None

    return env_name, transition_model, context_encoder, log_likelihood_model
