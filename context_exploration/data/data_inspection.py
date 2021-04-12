"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
from pathlib import Path

import numpy as np
from tqdm import tqdm

from context_exploration.data.dataset import NumpyDataset


def main():
    experience_dir = (
        Path(__file__)
        .resolve()
        .parent.parent.parent.joinpath(
            "data", "experience", "pendulum_quadrantactionfactorar2bd"
        )
    )
    n_rollouts = 1_000
    dataset = NumpyDataset(experience_dir, rollout_limit=n_rollouts)
    actions = []
    observations = []
    for idx in tqdm(range(n_rollouts)):
        data_item = dataset[idx]
        actions.append(data_item["action"][:-1])
        observations.append(data_item["observation"][:-1])
    actions = np.concatenate(actions)
    observations = np.concatenate(observations)
    all_data = np.concatenate((observations, actions), axis=-1)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=all_data.shape[-1], ncols=1, figsize=(10, 10))
    for dim in range(all_data.shape[-1]):
        ax[dim].hist(all_data[:, dim], bins=128)
    plt.show()


if __name__ == "__main__":
    main()
