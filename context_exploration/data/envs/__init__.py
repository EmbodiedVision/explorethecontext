"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
from context_exploration.data.envs.envs import *
from context_exploration.data.envs.mountain_car_env import *

ENVS = {
    "action_squash_1": [
        ActionSquashEnv,
        dict(action_squash_type="relu(1-abs(u))", disturbed=True),
    ],
    "action_squash_2": [
        ActionSquashEnv,
        dict(action_squash_type="relu(abs(u)-1)", disturbed=True),
    ],
    "action_squash_1_noisefree": [
        ActionSquashEnv,
        dict(action_squash_type="relu(1-abs(u))", disturbed=False),
    ],
    "action_squash_2_noisefree": [
        ActionSquashEnv,
        dict(action_squash_type="relu(abs(u)-1)", disturbed=False),
    ],
    "mountaincar": [MountainCarEnvRandomProfile, dict()],
}

_SPACE_TYPES = {
    "box_doublesided": "bd",
}

ENVS = dict(
    **ENVS,
    **{
        f"pendulum_quadrantactionfactorar{n}{s}": [
            PendulumEnvQuadrantActionFactor,
            {"action_repeat": n, "space_type": space_type},
        ]
        for n in [
            2,
        ]
        for space_type, s in _SPACE_TYPES.items()
    },
)


def make_env(env_name):
    return ENVS[env_name][0](**ENVS[env_name][1])
