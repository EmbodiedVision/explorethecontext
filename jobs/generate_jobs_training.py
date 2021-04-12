"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import itertools

COMMENT_PREFIX = "cr"
# named_config, optional: env_name
ENV_NAMES = {
    "asq1": ["action_squash", "action_squash_1"],
    "asq1_noisefree": ["action_squash", "action_squash_1_noisefree"],
    "asq2": ["action_squash", "action_squash_2"],
    "asq2_noisefree": ["action_squash", "action_squash_2_noisefree"],
    "mountaincar": ["mountaincar", "mountaincar"],
    "pendulum_bd": ["pendulum_bd", "pendulum_quadrantactionfactorar2bd"],
}

NP_KL_WEIGHT = [
    5,
]
SEED = [1, 2, 3]
POSWEIGHTS = [False, "relu"]

commands = []
for (
    environment_name,
    np_kl_weight,
    pos_weights,
    seed,
) in itertools.product(ENV_NAMES.keys(), NP_KL_WEIGHT, POSWEIGHTS, SEED):

    if environment_name.startswith("asq") and not pos_weights:
        continue

    pos_weights_cfg = {False: "no_positive_weights", "relu": ""}[pos_weights]
    pos_weights_str = {False: "False", "relu": "relu"}[pos_weights]
    command = (
        f"python -m context_exploration.train_model "
        f"--comment={COMMENT_PREFIX}_"
        f"s{seed}_"
        f"{environment_name}_"
        f"posweights_{pos_weights_str}_"
        f"npklw{np_kl_weight} "
        f"with "
        f"{ENV_NAMES[environment_name][0]}_cfg "
        f"env_name={ENV_NAMES[environment_name][1]} "
        f"{pos_weights_cfg} "
        f"kl_np_weight={np_kl_weight} "
        f"seed={seed}\n"
    )
    commands.append(command)

# write commands to job file
with open(f"jobs_training_{COMMENT_PREFIX}.txt", "w") as handle:
    handle.writelines(commands)
