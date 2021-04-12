"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import math
from contextlib import ContextDecorator
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from gym.spaces import Box
from torch.distributions import Normal

from context_exploration.data.envs.pendulum import NoActionRendererPendulumEnv
from context_exploration.data.wrappers import ActionRepeatWrapper, MaxDurationWrapper


class RandomSampleExcitationIterator:
    def __init__(self, env, seed):
        assert seed is not None
        self._env = env
        self._action_space = deepcopy(env.action_space)
        self._action_space.seed(seed)

    def __next__(self):
        return self._action_space.sample()


class RandomSampleExcitationController:
    def __init__(self, env):
        super(RandomSampleExcitationController, self).__init__()
        self._env = env

    def get_iterator(self, excitation_seed):
        return RandomSampleExcitationIterator(env=self._env, seed=excitation_seed)


class SampleActionMixin:
    def sample_action(self, *shape):
        if shape:
            action = np.stack(
                [self.action_space.sample() for _ in range(np.prod(np.array(shape)))]
            )
            action = action.reshape(*shape, *self.action_space.shape)
        else:
            action = self.action_space.sample()
        return action


class ParametrizedEnvNotInitialized(Exception):
    pass


class ParametrizedEnvAlreadyInitialized(Exception):
    pass


class ParametrizedEnvWrapper(gym.Env):
    def __init__(self, context_space):
        self._context_space = context_space
        self.env = None
        self.context = None
        self._last_seed = None

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def metadata(self):
        return self.env.metadata

    @property
    def context_dim(self):
        return self._context_space.shape[0]

    @property
    def context_space(self):
        return self._context_space

    @property
    def unwrapped(self):
        return self.env

    def initialize_context(self, seed):
        if self.context is not None:
            raise ParametrizedEnvAlreadyInitialized
        self._context_space.seed(seed)
        context = self._context_space.sample()
        self.env = self._construct_env(context)
        self.context = context
        self._last_seed = seed

    def _construct_env(self, context):
        raise NotImplementedError

    def release_context(self):
        self.context = None

    def _assert_initialized(self):
        if self.context is None:
            raise ParametrizedEnvNotInitialized

    def seed(self, seed=None):
        self._assert_initialized()
        return self.env.seed(seed)

    def reset(self, **kwargs):
        self._assert_initialized()
        return self.env.reset(**kwargs)

    def step(self, action):
        self._assert_initialized()
        return self.env.step(action)

    def render(self, mode="human"):
        self._assert_initialized()
        return self.env.render(mode)

    def close(self):
        self._assert_initialized()
        return self.env.close()

    def is_transition_informative(self, x, u, x_next):
        raise NotImplementedError


class SizeProperties:
    def __init__(self, state_dim, action_dim):
        self._state_dim = state_dim
        self._action_dim = action_dim

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def action_dim(self):
        return self._action_dim


class ActionSquashEnvBase(gym.Env, SizeProperties, SampleActionMixin):
    def __init__(self, action_squash_type: str, alpha: float, disturbed: bool):
        self.action_squash_type = action_squash_type
        state_dim = 2
        action_dim = 1

        self.action_space = Box(
            low=-2 * np.ones(action_dim), high=2 * np.ones(action_dim)
        )
        self.observation_space = Box(
            low=-np.inf * np.ones(state_dim), high=np.inf * np.ones(state_dim)
        )

        self.action_squash_type = action_squash_type
        self.alpha = alpha
        self.A = torch.Tensor([[0.8, 0.2], [-0.2, 0.8]])
        if disturbed:
            self.noise_covar = 0.01 ** 2
        else:
            self.noise_covar = 0
        self._step = None
        self.excitation_controller = RandomSampleExcitationController(self)
        self.rng = np.random.RandomState(None)
        SizeProperties.__init__(self, state_dim, action_dim)

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def reset(self, init_mode=None):
        if init_mode == "calibration":
            self.state = np.zeros(self.state_dim)
        elif init_mode is None:
            self.state = self.rng.randn(self.state_dim)
        else:
            raise ValueError("Invalid value for init_mode")
        self._step = 0
        return self.state

    def step(self, action: np.ndarray):
        if self._step is None:
            raise RuntimeError("Must reset environment.")

        self.state = self.step_single(
            torch.as_tensor(self.state).float(),
            torch.as_tensor(action).float(),
            torch.as_tensor(self.alpha).float(),
        ).numpy()

        done = False
        return self.state, 0, done, {}

    def render(self, mode: str = "human"):
        raise NotImplementedError

    def close(self):
        pass

    def _forward_mean(self, x: torch.Tensor, u: torch.Tensor, alpha: torch.Tensor):
        if x.device != self.A.device:
            self.A = self.A.to(x.device)
        B = torch.cat((alpha, torch.zeros_like(alpha)), dim=-1)
        B = B.unsqueeze(-1)  # add control input dimension
        if self.action_squash_type == "relu(1-abs(u))":
            squashed_action = F.relu(1.0 - torch.abs(u))
        elif self.action_squash_type == "relu(abs(u)-1)":
            squashed_action = F.relu(torch.abs(u) - 1)
        else:
            raise NotImplementedError
        x_contrib = torch.einsum("ij,...j->...i", self.A, x)
        u_contrib = torch.einsum("...ij,...j->...i", B, squashed_action)
        x_next_mean = x_contrib + u_contrib
        return x_next_mean

    def step_single(self, x: torch.Tensor, u: torch.Tensor, alpha: torch.Tensor):
        assert x.shape[-1] == self.state_dim
        assert u.shape[-1] == self.action_dim
        assert x.shape[:-1] == u.shape[:-1] == alpha.shape[:-1]
        assert alpha.shape[-1] == 1
        x_next_mean = self._forward_mean(x, u, alpha)
        noise = torch.randn_like(x) * math.sqrt(self.noise_covar)
        x_next = x_next_mean + noise
        return x_next

    def log_likelihood(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        x_next: torch.Tensor,
        alpha: torch.Tensor,
    ):
        assert x.shape[-1] == self.state_dim
        assert u.shape[-1] == self.action_dim
        assert x_next.shape[-1] == self.state_dim
        assert x.shape[:-1] == u.shape[:-1] == x_next.shape[:-1] == alpha.shape
        pred_mean = self._forward_mean(x, u, alpha)
        pred_distribution = Normal(
            pred_mean, torch.ones_like(pred_mean) * math.sqrt(self.noise_covar)
        )
        logll = pred_distribution.log_prob(x_next).sum(dim=-1)
        return logll

    def mse(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        x_next: torch.Tensor,
        alpha: torch.Tensor,
    ):
        assert x.shape[-1] == self.state_dim
        assert u.shape[-1] == self.action_dim
        assert x_next.shape[-1] == self.state_dim
        assert x.shape[:-1] == u.shape[:-1] == x_next.shape[:-1] == alpha.shape
        pred_mean = self._forward_mean(x, u, alpha)
        mse = ((pred_mean - x_next) ** 2).sum(dim=-1)
        return mse

    def sample_calibration_initial_state(self, *shape):
        return np.zeros(shape + (self.state_dim,))


class ActionSquashEnv(ParametrizedEnvWrapper, SizeProperties, SampleActionMixin):
    def __init__(self, action_squash_type: str, disturbed: bool):
        self._action_squash_type = action_squash_type
        self._disturbed = disturbed
        context_space = Box(low=-1 * np.ones(1), high=1 * np.ones(1))
        self.max_duration = 100
        super(ActionSquashEnv, self).__init__(context_space)
        state_dim = 2
        action_dim = 1
        SizeProperties.__init__(self, state_dim, action_dim)

    @property
    def excitation_controller(self):
        return self.env.excitation_controller

    def _construct_env(self, context):
        alpha = context
        env = MaxDurationWrapper(
            ActionSquashEnvBase(
                action_squash_type=self._action_squash_type,
                alpha=alpha,
                disturbed=self._disturbed,
            ),
            max_duration=self.max_duration,
        )
        return env

    def is_transition_informative(
        self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray
    ):
        assert u.shape[-1] == 1
        if self._action_squash_type == "relu(1-abs(u))":
            is_informative = np.abs(u.squeeze(-1)) <= 1
        elif self._action_squash_type == "relu(abs(u)-1)":
            is_informative = np.abs(u.squeeze(-1)) >= 1
        else:
            raise NotImplementedError
        return is_informative

    def get_domain(self):
        return 0


class ObservationDisturbanceWrapper(gym.Wrapper):
    """
    Add disturbance to observations.
    Can be disabled by
    ```
    with env.inhibit_disturbance:
        ...
    ```
    """

    def __init__(self, env, disturbance_fcn, enabled):
        super(ObservationDisturbanceWrapper, self).__init__(env)
        self.disturbance_fcn = disturbance_fcn
        self.enabled = enabled
        self.inhibit_disturbance = ObservationDisturbanceWrapper.InhibitionDecorator(
            self
        )
        self._inhibit = False

    @property
    def apply_disturbance(self):
        return self.enabled and not self._inhibit

    class InhibitionDecorator(ContextDecorator):
        def __init__(self, parent):
            self.obj = parent

        def __enter__(self):
            self.obj._inhibit = True

        def __exit__(self, *exc):
            self.obj._inhibit = False

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs_disturbed = self.disturbance_fcn(obs) if self.apply_disturbance else obs
        return obs_disturbed

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_disturbed = self.disturbance_fcn(obs) if self.apply_disturbance else obs
        return obs_disturbed, reward, done, info


class PendulumEnv(ParametrizedEnvWrapper, SizeProperties, SampleActionMixin):
    def __init__(self, disturbed):
        self.max_duration = 100
        self.disturbed = disturbed
        state_dim = 3
        action_dim = 1
        # pole_mass_range = (0.5, 2)  # default 1.0
        # pole_length_range = (0.5, 2)  # default 1.0
        context_space = Box(low=np.array([0.5, 0.5]), high=np.array([2.0, 2.0]))
        self._disturbance_rng = None
        self.excitation_controller = RandomSampleExcitationController(self)
        ParametrizedEnvWrapper.__init__(self, context_space)
        SizeProperties.__init__(self, state_dim, action_dim)

    def reset(self, init_mode=None):
        if init_mode == "calibration":
            return super().reset(init_type="bottom")
        elif init_mode is None:
            return super().reset()
        else:
            raise ValueError("Invalid value for init_mode")

    def _construct_env(self, context):
        m_pole, l_pole = context
        environment_kwargs = {"m": m_pole, "l": l_pole}
        env = NoActionRendererPendulumEnv(**environment_kwargs, init_type="random")
        env = MaxDurationWrapper(env, self.max_duration)
        env = ObservationDisturbanceWrapper(
            env, self.disturbance_fcn, enabled=self.disturbed
        )
        return env

    def seed(self, seed=None):
        super().seed(seed)
        self._disturbance_rng = np.random.RandomState(seed)

    def disturbance_fcn(self, observation):
        # observation: [angle_cos angle_sin vel_theta]
        angle = np.arctan2(observation[1], observation[0])
        # angle in [-pi .. pi]
        theta_dot = observation[2]
        assert np.isclose(np.cos(angle), observation[0])
        assert np.isclose(np.sin(angle), observation[1])
        # activate disturbance when not in upright position and moving slowly
        if not (-np.pi / 4 <= angle <= np.pi / 4 and -2 <= theta_dot <= 2):
            # Disturb angular velocity by a std of 0.2 deg/sec (range: -8 .. +8)
            theta_dot += self._disturbance_rng.randn() * 0.2
            # Disturb angle by a std of 5 deg
            angle += self._disturbance_rng.randn() * np.deg2rad(5)
            observation = np.array([np.cos(angle), np.sin(angle), theta_dot])
        return observation

    def is_transition_informative(self, x, u, x_next):
        angle = np.arctan2(x[..., 1], x[..., 0])
        vel = x[..., 2]
        return np.logical_and(
            np.logical_and(angle >= -np.pi / 4, angle <= np.pi / 4),
            np.logical_and(vel >= -2, vel <= 2),
        )


class NoActionRendererPendulumQuadrantActionFactor(NoActionRendererPendulumEnv):
    def __init__(
        self,
        quadrant_action_factor,
        max_torque=2.0,
        max_speed=8.0,
        init_type="standard",
    ):
        super(NoActionRendererPendulumQuadrantActionFactor, self).__init__(
            max_torque=max_torque, max_speed=max_speed, init_type=init_type
        )
        self.quadrant_action_factor = quadrant_action_factor

    def get_domain(self):
        th, thdot = self.state
        norm_angle = NoActionRendererPendulumEnv.angle_normalize(th)
        # norm_angle is in [-pi, pi]
        if -math.pi <= norm_angle < -math.pi / 2:
            quadrant = 3
        elif -math.pi / 2 <= norm_angle < 0:
            quadrant = 2
        elif 0 <= norm_angle < math.pi / 2:
            quadrant = 1
        elif math.pi / 2 <= norm_angle <= math.pi:
            quadrant = 0
        else:
            raise ValueError
        return quadrant

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        norm_angle = NoActionRendererPendulumEnv.angle_normalize(th)

        # norm_angle is in [-pi, pi]
        if -math.pi <= norm_angle < -math.pi / 2:
            action_factor = self.quadrant_action_factor[3]
        elif -math.pi / 2 <= norm_angle < 0:
            action_factor = self.quadrant_action_factor[2]
        elif 0 <= norm_angle < math.pi / 2:
            action_factor = self.quadrant_action_factor[1]
        elif math.pi / 2 <= norm_angle <= math.pi:
            action_factor = self.quadrant_action_factor[0]
        else:
            raise ValueError

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        u = u * action_factor

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


class DiscreteQuadrantPropertySpace(gym.Space):
    def __init__(self, per_quadrant_set):
        super(DiscreteQuadrantPropertySpace, self).__init__(
            shape=(4,), dtype=per_quadrant_set.dtype
        )
        self.per_quadrant_set = per_quadrant_set

    def sample(self):
        return self.np_random.choice(self.per_quadrant_set, 4)


class HierarchicalBoxSpace(gym.Space):
    def __init__(self, boxes, sampling_weights):
        first_box = boxes[0]
        assert len(boxes) == len(sampling_weights)
        sampling_weights = np.array(sampling_weights)
        assert np.sum(sampling_weights) == 1
        for box in boxes[1:]:
            assert box.dtype == first_box.dtype
            assert box.shape == first_box.shape
        self.sampling_weights = sampling_weights
        self.boxes = boxes
        super(HierarchicalBoxSpace, self).__init__(
            shape=first_box.shape, dtype=first_box.dtype
        )

    def seed(self, seed=None):
        super().seed(seed)
        for box in self.boxes:
            box.seed(seed)

    def sample(self):
        box_samples = np.stack([box.sample() for box in self.boxes])
        choices = self.np_random.choice(
            np.arange(len(self.boxes)), self.shape[0], p=self.sampling_weights
        )
        samples = np.take_along_axis(box_samples, choices[None, :], axis=0).squeeze(0)
        return samples


class PendulumEnvQuadrantActionFactor(
    ParametrizedEnvWrapper, SizeProperties, SampleActionMixin
):
    def __init__(self, action_repeat, space_type):
        self.action_repeat = action_repeat
        self.max_duration = 100
        state_dim = 3
        action_dim = 1
        # with a max_factor of 1.5, the pendulum can still not be
        # driven directly upwards
        if space_type == "box_singlesided":
            context_space = Box(low=np.array([0.5] * 4), high=np.array([2] * 4))
        elif space_type == "box_doublesided":
            context_space = HierarchicalBoxSpace(
                [
                    Box(low=np.array([-2] * 4), high=np.array([-0.5] * 4)),
                    Box(low=np.array([0.5] * 4), high=np.array([2] * 4)),
                ],
                sampling_weights=[0.5, 0.5],
            )
        elif space_type == "discrete_singlesided":
            context_space = DiscreteQuadrantPropertySpace(np.array([0.5, 2]))
        elif space_type == "discrete_doublesided":
            context_space = DiscreteQuadrantPropertySpace(np.array([-2, -0.5, 0.5, 2]))
        else:
            raise ValueError
        self.excitation_controller = RandomSampleExcitationController(self)
        ParametrizedEnvWrapper.__init__(self, context_space)
        SizeProperties.__init__(self, state_dim, action_dim)

    def get_domain(self):
        return self.env.get_domain()

    def reset(self, init_mode=None):
        if init_mode == "calibration":
            return super().reset(init_type="bottom")
        elif init_mode is None:
            return super().reset()
        else:
            raise ValueError("Invalid value for init_mode")

    def _construct_env(self, context):
        environment_kwargs = {"quadrant_action_factor": context}
        env = NoActionRendererPendulumQuadrantActionFactor(
            **environment_kwargs, init_type="random"
        )
        env = ActionRepeatWrapper(env, action_repeat=self.action_repeat)
        env = MaxDurationWrapper(env, self.max_duration)
        return env

    def seed(self, seed=None):
        super().seed(seed)

    def is_transition_informative(self, x, u, x_next):
        return np.ones(*x.shape[:-1])


def sample_pendulum_envs():
    env = PendulumEnvQuadrantActionFactor(action_repeat=2, space_type="box_doublesided")
    for idx in range(200):
        env.initialize_context(idx)
        print(env.context)
        env.release_context()


if __name__ == "__main__":
    sample_pendulum_envs()

    import time

    env = PendulumEnvQuadrantActionFactor(action_repeat=2, space_type="box_singlesided")
    env.initialize_context(21)
    env.seed(42)
    action_iterator = env.excitation_controller.get_iterator(excitation_seed=41)
    env.reset()
    for _ in range(50):
        action = next(action_iterator)
        obs, _, _, _ = env.step(action)
        env.render(mode="human")
        print(obs)
        time.sleep(0.1)
    env.close()
