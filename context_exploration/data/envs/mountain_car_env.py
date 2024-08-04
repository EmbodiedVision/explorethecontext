"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import gym
import numpy as np
from gym.spaces import Box
from gym.utils import seeding

from context_exploration.data.envs.envs import (
    MaxDurationWrapper,
    ParametrizedEnvWrapper,
    RandomSampleExcitationController,
    SampleActionMixin,
    SizeProperties,
)


class ProfileMountainCarEnv(gym.Env):
    def __init__(self, profile_fcn=None, grad_fcn=None):
        # the profile should range from 0 to 1
        # in the boundaries of -1/1
        if profile_fcn is None:
            self.profile_fcn = lambda x: 0.5 + 0.45 * np.sin(np.pi * x)
            self.grad_fcn = lambda x: 0.25 * np.pi * np.cos(np.pi * x)
        else:
            assert grad_fcn is not None
            self.profile_fcn = profile_fcn
            self.grad_fcn = grad_fcn

        x = np.linspace(-1, 1, 500)
        profile = self.profile_fcn(x)
        self.max_height = np.max(profile)

        self.dt = 0.05

        self.g = -10
        self.fric = 0.001
        self.state = None
        self.max_u = 3

        self.action_space = Box(
            low=-self.max_u, high=self.max_u, shape=(1,), dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, init_mode="random"):
        if init_mode == "calibration":
            self.state = np.array([0, 0])
        elif init_mode == "random":
            self.state = np.array(
                [self.np_random.uniform(-0.8, 0.8), self.np_random.uniform(-2, 2)]
            )
        else:
            raise ValueError(f"Unknown init_mode {init_mode}")
        return self.state

    def get_reward(self, state, action):
        height = self.profile_fcn(state[..., 0])
        costs = (height - self.max_height) ** 2
        return -costs

    def step(self, action: np.ndarray):
        u = action[0]
        u = np.clip(u, -self.max_u, self.max_u)

        x, xdot = self.state

        reward = self.get_reward(self.state, action)

        grad_x = self.grad_fcn(x)
        # euler integration
        for step in range(2):
            grad_angle = np.arctan(grad_x)
            tang_accel = self.g * np.sin(grad_angle) + u
            max_fric_accel = np.abs(self.g * np.cos(grad_angle) * self.fric)
            abs_fric_accel = min(max_fric_accel, np.abs(tang_accel))
            accel_x = (tang_accel - np.sign(xdot) * abs_fric_accel) * np.cos(grad_angle)
            x_new = x + self.dt * xdot + 0.5 * accel_x * self.dt ** 2
            grad_x = 0.5 * grad_x + 0.5 * self.grad_fcn(x_new)

        xdot_new = xdot + accel_x * self.dt

        if x_new >= 1:
            x_new = 1
            xdot_new = 0
        if x_new <= -1:
            x_new = -1
            xdot_new = 0

        obs = np.array([x_new, xdot_new])
        self.state = obs
        done = False
        info = {}
        return obs, reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class MountainCarProfileContextSpace(gym.Space):
    def __init__(self):
        super(MountainCarProfileContextSpace, self).__init__(
            shape=(0,), dtype=object
        )

    def sample(self):
        n_points = self.np_random.choice(np.arange(2, 7 + 1))
        locations = self.np_random.uniform(-1.5, 1.5, n_points)
        heights = self.np_random.uniform(0.1, 0.3, n_points)
        widths = self.np_random.uniform(0.1, 0.5, n_points)
        locations = np.concatenate((locations, np.array([-1, 1])))
        heights = np.concatenate((heights, np.array([0.5, 0.5])))
        widths = np.concatenate((widths, np.array([0.3, 0.3])))
        return np.stack((locations, heights, widths))


class MountainCarEnvRandomProfile(
    ParametrizedEnvWrapper, SizeProperties, SampleActionMixin
):
    def __init__(self):
        self.max_duration = 100
        state_dim = 2
        action_dim = 1
        context_space = MountainCarProfileContextSpace()
        self.excitation_controller = RandomSampleExcitationController(self)
        ParametrizedEnvWrapper.__init__(self, context_space)
        SizeProperties.__init__(self, state_dim, action_dim)

    @property
    def profile_fcn(self):
        return self.env.profile_fcn

    @property
    def grad_fcn(self):
        return self.env.grad_fcn

    def get_domain(self):
        return 0

    def reset(self, init_mode="random"):
        return super().reset(init_mode=init_mode)

    @staticmethod
    def profile_from_context(context_sample):
        locations = context_sample[0, :]
        heights = context_sample[1, :]
        widths = context_sample[2, :]
        profile_fcn = lambda x: sum(
            h * np.exp(-0.5 * ((x - l) ** 2 / w ** 2))
            for l, h, w in zip(locations, heights, widths)
        )
        grad_fcn = lambda x: sum(
            h * np.exp(-0.5 * ((x - l) ** 2 / w ** 2)) * ((-0.5 / w ** 2) * 2 * (x - l))
            for l, h, w in zip(locations, heights, widths)
        )
        return profile_fcn, grad_fcn

    def _construct_env(self, context):
        profile_fcn, grad_fcn = MountainCarEnvRandomProfile.profile_from_context(
            context
        )
        env = ProfileMountainCarEnv(profile_fcn=profile_fcn, grad_fcn=grad_fcn)
        env = MaxDurationWrapper(env, self.max_duration)
        return env

    def get_reward(self, state, action):
        return self.env.get_reward(state, action)

    def seed(self, seed=None):
        super().seed(seed)

    def is_transition_informative(self, x, u, x_next):
        return np.ones(*x.shape[:-1])


def run_mountaincar():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1)
    x = np.linspace(-1, 1, 200)
    env = ProfileMountainCarEnv()
    profile = env.profile_fcn(x)
    ax.plot(x, profile)

    env.reset()
    env.state = np.array([-0.5, 0])
    for idx in range(50):
        if idx < 15:
            action = -3
        else:
            action = 3
        obs, _, _, _ = env.step(np.array([action]))
        print(obs[1])
        ax.scatter(obs[0], env.profile_fcn(obs[0]))
        plt.savefig(f"mountaincar/mountaincar_{idx}.png")
    plt.show()
    env.close()


def sample_profiles():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1)
    context_space = MountainCarProfileContextSpace()
    profile_fcn, grad_fcn = MountainCarEnvRandomProfile.profile_from_context(
        context_space.sample()
    )
    x = np.linspace(-1, 1, 200)
    profile = profile_fcn(x)
    grad = grad_fcn(x)
    ax.plot(x, profile)
    ax.plot(x, grad)
    # ax.set_ylim(0, 1)
    plt.show()


def run_random_mountaincar():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1)
    x = np.linspace(-1, 1, 200)
    env = MountainCarEnvRandomProfile()
    env.initialize_context(42)
    profile = env.profile_fcn(x)
    ax.plot(x, profile)

    env.reset()
    for idx in range(50):
        obs, _, _, _ = env.step(env.action_space.sample())
        print(obs[1])
        ax.scatter(obs[0], env.profile_fcn(obs[0]))
        plt.savefig(f"mountaincar/random_mountaincar_{idx}.png")
    plt.show()
    env.close()


if __name__ == "__main__":
    run_random_mountaincar()
