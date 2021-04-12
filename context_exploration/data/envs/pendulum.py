"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
__all__ = ["RandomInitPendulumEnv", "NoActionRendererPendulumEnv"]


import warnings

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class RandomInitPendulumEnv(gym.Wrapper):
    def __init__(self, **kwargs):
        pendulum_env = NoActionRendererPendulumEnv(**kwargs)
        super(RandomInitPendulumEnv, self).__init__(pendulum_env)

    def reset(self):
        high = np.array([np.pi, self.env.max_speed])
        self.env.state = self.env.np_random.uniform(low=-high, high=high)
        self.env.last_u = None
        return self.env._get_obs()

    def _get_obs(self):
        return self.env._get_obs()


class NoActionRendererPendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(
        self,
        g=10.0,
        m=1.0,
        l=1.0,
        action_factor=1.0,
        render_pole_length=1,
        render_pole_width=1,
        render_pole_center=(0, 0),
        max_torque=2.0,
        max_speed=8.0,
        init_type="standard",
    ):
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.dt = 0.05
        self.g = g
        self.m = m
        self.l = l
        self.action_factor = action_factor
        self.viewer = None
        self.render_pole_length = render_pole_length
        self.render_pole_width = render_pole_width
        self.render_pole_center = render_pole_center
        self.init_type = init_type

        high = np.array([1.0, 1.0, self.max_speed])
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        u = self.action_factor * u

        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        norm_angle = NoActionRendererPendulumEnv.angle_normalize(th)
        costs = norm_angle ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        newthdot = (
            thdot
            + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l ** 2) * u) * dt
        )
        newth = th + newthdot * dt
        newthdot = np.clip(
            newthdot, -self.max_speed, self.max_speed
        )  # pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self, init_type=None):
        if init_type is None:
            init_type = self.init_type
        if init_type is not None and init_type != self.init_type:
            warnings.warn(
                f"Overwriting self.init_type={self.init_type} with init_type={init_type}"
            )
        if init_type == "standard":
            high = np.array([np.pi, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
        elif init_type == "random":
            high = np.array([np.pi, self.max_speed])
            self.state = self.np_random.uniform(low=-high, high=high)
        elif init_type == "bottom":
            high = np.array([0.05, 0.05])
            self.state = self.np_random.uniform(low=-high, high=high)
            self.state[0] += np.pi
        else:
            raise ValueError
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    # Also provide get_obs in a non-private function to access when wrapped
    def get_obs(self):
        return self._get_obs()

    def render(self, mode="human"):

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(
                1 * self.render_pole_length, 0.2 * self.render_pole_width
            )
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            axle.add_attr(self.pole_transform)
            self.viewer.add_geom(axle)

        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        self.pole_transform.set_translation(*self.render_pole_center)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    @staticmethod
    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi
