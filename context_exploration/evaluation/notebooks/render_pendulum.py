import numpy as np

from context_exploration.data.envs.pendulum import NoActionRendererPendulumEnv

env = NoActionRendererPendulumEnv()
env.seed(42)
env.reset()
rgb_array = env.render(mode="rgb_array")
np.savez("pendulum_rendering.npz", rgb_array)
env.close()
