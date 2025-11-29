import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete


class FlattenMultiDiscreteAction(gym.Wrapper):
    """
    Wrap an env with MultiDiscrete action space and expose a single Discrete(n)
    action space where n = prod(nvec). Steps:
      - encode([a0, a1, ..., ak]) -> id in [0, n)
      - decode(id) -> original MultiDiscrete vector

    Observation/reward/info are pass-through.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        orig = env.action_space
        if not isinstance(orig, MultiDiscrete):
            raise TypeError(
                "FlattenMultiDiscreteAction requires a MultiDiscrete action space."
            )

        # nvec: number of choices per dimension, e.g. [A, T, X, Y]
        self._nvec = np.array(orig.nvec, dtype=np.int64)

        # radices[i] = prod(nvec[i+1:]) so that
        # id = sum( a[i] * radices[i] )
        self._radices = np.empty_like(self._nvec)
        prod = 1
        for i in range(len(self._nvec) - 1, -1, -1):
            self._radices[i] = prod
            prod *= self._nvec[i]

        self._n_actions = int(prod)
        self.action_space = Discrete(self._n_actions)

    def encode(self, a_vec: np.ndarray | list[int]) -> int:
        a_vec = np.asarray(a_vec, dtype=np.int64)
        return int(np.sum(a_vec * self._radices))

    def decode(self, action_id: int) -> np.ndarray:
        action_id = int(action_id)
        a = np.empty_like(self._nvec)
        rem = action_id
        for i in range(len(self._nvec)):
            a[i] = rem // self._radices[i]
            rem = rem % self._radices[i]
        return a

    def step(self, action):
        decoded = self.decode(action)
        return self.env.step(decoded)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self):
        # pass-through for video wrappers
        return self.env.render()

