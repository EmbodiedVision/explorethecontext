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
import pickle as pkl
import uuid
import warnings
from collections import namedtuple
from io import BytesIO
from os.path import join

import gym
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from context_exploration.utils import QuadraticResize


class LazyWrapper(gym.Wrapper):
    def __init__(self, env):
        # super().__init__ is not called here on purpose
        self.env = env

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


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat):
        super(ActionRepeatWrapper, self).__init__(env)
        self.action_repeat = action_repeat

    @property
    def dt(self):
        return self.env.dt * self.action_repeat

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.action_repeat and not done:
            observ, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return observ, total_reward, done, info


class MaxDurationWrapper(gym.Wrapper):
    def __init__(self, env, max_duration):
        super(MaxDurationWrapper, self).__init__(env)
        self.max_duration = max_duration
        self._step = None

    def step(self, action):
        if self._step is None:
            raise RuntimeError("Must reset environment.")
        observ, reward, done, info = self.env.step(action)
        self._step += 1
        if done or self._step >= self.max_duration:
            done = True
            self._step = None
        return observ, reward, done, info

    def reset(self, **kwargs):
        self._step = 0
        return self.env.reset(**kwargs)


def process_rendering(env, rendering_size, as_png):
    screen = env.render(mode="rgb_array")
    pil_image = Image.fromarray(screen)
    pil_image = QuadraticResize(rendering_size)(pil_image)
    if as_png:
        with BytesIO() as byte_io:
            pil_image.save(byte_io, "PNG")
            return byte_io.getvalue()
    else:
        return np.array(pil_image)


RolloutFilenames = namedtuple(
    "RolloutFilenames", ["rollout_filename", "rendering_filename", "metadata_filename"]
)


class CaptureWrapper(LazyWrapper):
    def __init__(
        self,
        env,
        rollout_uuid=None,
        save_renderings=False,
        rendering_size=(64, 64),
        rendering_as_png=False,
        process_rendering_fcn=None,
    ):
        super(CaptureWrapper, self).__init__(env)
        self.save_renderings = save_renderings
        self.rendering_size = rendering_size
        self.rendering_as_png = rendering_as_png
        if process_rendering_fcn is None:
            self._process_rendering = lambda: process_rendering(
                env, rendering_size, rendering_as_png
            )
        else:
            self._process_rendering = lambda: process_rendering_fcn(
                env.render(mode="rgb_array")
            )
        self.rollout_uuid = rollout_uuid
        self._in_capture_mode = False

    def reset(self, **kwargs):
        if self._in_capture_mode:
            raise RuntimeError("env.reset() must only be called once.")
        obs = self.env.reset()
        self._action = []
        self._reward = [np.nan]
        self._done = [False]
        self._info = [{}]
        self._observation = [obs]
        if self.save_renderings:
            self._rendering = [self._process_rendering()]
        self._in_capture_mode = True
        return obs

    def step(self, action):
        if not self._in_capture_mode:
            raise RuntimeError("Must reset environment.")
        obs, reward, done, info = self.env.step(action)
        self._action.append(action)
        self._reward.append(reward)
        self._done.append(done)
        self._info.append(info)
        self._observation.append(obs)
        if self.save_renderings:
            self._rendering.append(self._process_rendering())
        return obs, reward, done, info

    def close(self):
        if not self._in_capture_mode:
            raise RuntimeError("Cannot close environment; env.reset() was not called.")
        # append 0 action to last step
        self._action.append(self._action[-1] * 0)
        # bring first reward to right shape
        self._reward[0] = np.ones_like(self._reward[1]) * np.nan
        rollout = {
            "action": np.stack(self._action),
            "reward": np.stack(self._reward),
            "info": self._info,
            "done": self._done,
            "observation": self._observation,
        }
        data_dict = {
            "id": self.rollout_uuid,
            "meta": {"rollout_length": len(self._done)},
            "rollout": rollout,
        }
        # save renderings
        if self.save_renderings:
            data_dict["rendering"] = {
                "rendering": self._rendering,
                "rendering_as_png": self.rendering_as_png,
            }
        self.env.close()
        return data_dict
