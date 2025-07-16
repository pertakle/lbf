import gymnasium as gym
import numpy as np


class RewardShapingWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env, args):
        gym.utils.RecordConstructorArgs.__init__(self)
        super().__init__(env)
        self._args = args
        self._last_distances = np.zeros(args.players + args.foods)

    def reset(self, *args, **kwargs):
        state, info = self.env.reset(*args, **kwargs)
        self._last_distances = self._distances(state)
        return state, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)

        new_distances = self._distances(next_state)
        shaped_reward = self._shaped_reward(new_distances - self._last_distances)

        self._last_distances = new_distances
        # return next_state, reward + shaped_reward * (reward == 0), terminated, truncated, info
        done = terminated or truncated
        return next_state, 2 * reward + (1 - done) * shaped_reward, terminated, truncated, info

    def _shaped_reward(self, distances_delta):
        return -(distances_delta.mean() / self._args.env_size) / 1

    def _distances(self, state):
        reshaped_state = state.reshape(-1, 3)
        foods = reshaped_state[:self._args.foods, :2]
        players = reshaped_state[self._args.foods:, :2]

        dists = np.abs(players[:, None, :] - foods[None, :, :]).sum(-1).min(-1)
        return dists


class NpWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env, args):
        gym.utils.RecordConstructorArgs.__init__(self)
        super().__init__(env)
        self._args = args

        # 3-hot
        self._observation_space = gym.spaces.MultiBinary(
            [(self._args.players + self._args.foods) * (self._args.env_size * 2 + self._args.players + 1)]
        )

        # tripplets
        # self._observation_space = gym.spaces.Box(
            # np.zeros((self._args.players + self._args.foods) * 3, dtype=np.float32),
            # np.array([1, 1, 1] * (self._args.players + self._args.foods), dtype=np.float32),  # normalized
            # # np.array([self._args.env_size, self._args.env_size, self._args.players] * (self._args.players + self._args.foods), dtype=np.float32),
        # )
        self._action_space = gym.spaces.MultiDiscrete([6]*self._args.players)

    def _full_observation_tripples(self, obs: tuple[np.ndarray, ...]) -> np.ndarray:
        """Original observation is a tuple of flattened tripples (row, col, level):
        food 1, food 2, ..., food F, agent i, other agents' positions...

        Returns: ndarray (foods + players, 3)

        Already eaten food is reprezented by zeros.
        We support only fully observable version of the environment.
        """
        foods = obs[0].astype(np.int64).reshape(-1, 3)[:self._args.foods]
        padded_foods = np.zeros([self._args.foods, 3], dtype=np.int64)
        padded_foods[:len(foods)] = foods
        positions = np.array([ob.astype(np.int64).reshape(-1, 3)[self._args.foods] for ob in obs])
        new_obs = np.concat([padded_foods, positions], axis=0)
        return new_obs

    def _three_hot_observation(self, obs: np.ndarray) -> np.ndarray:
        """Makes three-hot reprezentation.
        Transforms each value in the tripplet into one-hot encoding.

        Returns: ndarray, ((foods + players), (env_size * 2 + players))
        """
        new_obs = np.zeros(((self._args.foods + self._args.players), (self._args.env_size * 2 + self._args.players + 1)), dtype=np.int64)
        _rang = np.arange(self._args.foods + self._args.players)
        new_obs[_rang, obs[:, 0]] = 1
        new_obs[_rang, self._args.env_size + obs[:, 1]] = 1
        new_obs[_rang, 2*self._args.env_size + obs[:, 2]] = 1
        new_obs[:self._args.foods] *= obs[:self._args.foods, [2]] > 0
        return new_obs

    def observation(self, obs):
        new_obs = self._full_observation_tripples(obs)
        # print(new_obs)

        # new_obs = new_obs.astype(np.float64)
        # new_obs[:, :2] /= self._args.env_size
        # new_obs[:, 2] /= self._args.players

        new_obs = self._three_hot_observation(new_obs)
        # print(new_obs)
        # print()
        new_obs = new_obs.reshape(-1)
        return new_obs

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(self, action):
        # original env wants tuple instead of ndarray, well done
        tuple_action = tuple(a.item() for a in action)

        obs, rewards, terminated, truncated, info = super().step(tuple_action)
        obs = self.observation(obs)
        reward = np.sum(rewards)  # type: ignore
        return obs, reward, terminated, truncated, info
        

