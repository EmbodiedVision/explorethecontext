"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import itertools
import json
from pathlib import Path

from tqdm import tqdm

COMMENT_PREFIX = "cr"

# Toy problem (action squash) is calibrated after 50k training steps,
# more involved environments after 100k training steps.
# Calibration rollout length is {1, 2, 3} for toy problem,
# {15, 30, 50} for pendulum envs and {20, 50, 100} for MountainCar

WITH_ABLATIONS = False

ASQ_OPTIONS = {
    "rollout_length_list": [1, 2, 3],
    "ckpt": "50000_best",
    "validation_rollout_length": 50,
    "num_rollout_list": [
        1,
    ],
    "cemph_list": [
        20,
    ],
}
PENDULUM_OPTIONS = {
    "rollout_length_list": [15, 30, 50]
    if WITH_ABLATIONS
    else [
        30,
    ],
    "ckpt": "100000_best",
    "validation_rollout_length": 50,
    "num_rollout_list": [1, 3]
    if WITH_ABLATIONS
    else [
        1,
    ],
    "cemph_list": [
        20,
    ],
}
MOUNTAINCAR_OPTIONS = {
    "rollout_length_list": [20, 50, 100]
    if WITH_ABLATIONS
    else [
        50,
    ],
    "ckpt": "100000_best",
    "validation_rollout_length": 50,
    "num_rollout_list": [1, 3]
    if WITH_ABLATIONS
    else [
        1,
    ],
    "cemph_list": [
        30,
    ],
}
ENVIRONMENTS = {
    "action_squash_1": ASQ_OPTIONS,
    "action_squash_1_noisefree": ASQ_OPTIONS,
    "action_squash_2": ASQ_OPTIONS,
    "action_squash_2_noisefree": ASQ_OPTIONS,
    "pendulum_quadrantactionfactorar2bd": PENDULUM_OPTIONS,
    "mountaincar": MOUNTAINCAR_OPTIONS,
}

calibration_jobs = []
# Iterate over all available runs
RUN_DIR = Path(__file__).resolve().parent.parent.joinpath("experiments", "train_model")
dir_names = sorted(list(RUN_DIR.iterdir()))
for dir_name in dir_names:
    if not dir_name.joinpath("config.json").is_file():
        continue
    with open(dir_name.joinpath("config.json"), "r") as handle:
        config = json.load(handle)
    env_name = config["env_name"]

    print(f"Found {env_name} at {dir_name.name}")

    num_rollout_list = ENVIRONMENTS[env_name]["num_rollout_list"]
    rollout_length_list = ENVIRONMENTS[env_name]["rollout_length_list"]
    run_id = dir_name.name
    ckpt = ENVIRONMENTS[env_name]["ckpt"]
    validation_rollout_length = ENVIRONMENTS[env_name]["validation_rollout_length"]
    cemph_list = ENVIRONMENTS[env_name]["cemph_list"]

    for num_rollouts, rollout_length, cemph in itertools.product(
        num_rollout_list, rollout_length_list, cemph_list
    ):
        command = (
            "python -m context_exploration.evaluation.evaluate_calibration "
            f"{run_id} "
            f"--ckpt {ckpt} "
            f"--nenvinits {num_rollouts} "
            f"--cemph {cemph} "
            f"--rlength {rollout_length} "
            f"--vlength {validation_rollout_length}\n"
        )
        calibration_jobs.append(command)

# write commands to job file
with open(f"jobs_calibration_{COMMENT_PREFIX}.txt", "w") as handle:
    handle.writelines(calibration_jobs)
