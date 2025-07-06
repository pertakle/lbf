import numpy as np
import torch
import gymnasium as gym

import argparse
import time
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="model", help="path to the model file")
parser.add_argument("--hidden_size", type=int, default=50, help="size of hidden layer")
parser.add_argument("--num_envs", type=int, default=64, help="number of parallel training envs")
parser.add_argument("--ep_limit", type=int, default=20, help="number of parallel training envs")
parser.add_argument("--eval_each", type=int, default=100, help="evaluation period")
parser.add_argument("--eval_for", type=int, default=10, help="number of eval_episodes")
parser.add_argument("--render_each", type=int, default=10, help="render eval eps")

parser.add_argument("--env_size", type=int, default=5, help="size of the envirnonment")
parser.add_argument("--players", type=int, default=3, help="number of players")
parser.add_argument("--foods", type=int, default=2, help="number of players")

parser.add_argument("--train_steps", type=int, default=1_000_000, help="number of training steps")
parser.add_argument("--steps_per_update", type=int, default=10, help="number of steps in the env per update")
parser.add_argument("--batch_size", type=int, default=32, help="size of a single batch")
parser.add_argument("--epochs", type=int, default=2, help="number of epochs per training")
parser.add_argument("--lr", type=float, default=3e-4, help="number of steps in the env per update")
parser.add_argument("--gamma", type=float, default=0.99, help="number of steps in the env per update")
parser.add_argument("--lambd", type=float, default=0.95, help="gae trace lambda")
parser.add_argument("--tau", type=float, default=0.01, help="gae trace lambda")
parser.add_argument("--clip_eps", type=float, default=0.25, help="ppo clip")
parser.add_argument("--entropy_reg", type=float, default=0.05, help="entropy regularization")
parser.add_argument("--clip_grad_norm", type=float, default=10000., help="gradient clipping")



class TrajectoryBuffer:
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.last_next_states = None
        self.prev_dones = None
        
    def append(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        actions_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_states: np.ndarray,
        prev_dones: np.ndarray
    ) -> None:
        self.states.append(states)
        self.actions.append(actions)
        self.action_probs.append(actions_probs)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.values.append(values)
        self.last_next_states = next_states
        if self.prev_dones is None:
            self.prev_dones = prev_dones

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.action_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.last_next_states = None
        self.prev_dones = None

    def extended_states(self) -> np.ndarray:
        assert self.last_next_states is not None
        return np.concat([self.states, self.last_next_states[None]], axis=0)

    def extended_dones(self) -> np.ndarray:
        assert self.prev_dones is not None
        return np.concat([self.prev_dones[None], self.dones], axis=0)

    def get_prev_dones(self) -> np.ndarray:
        return self.extended_dones()[:-1]


def torch_init_with_orthogonal_and_zeros(module):
    """Initialize weights of a PyTorch module with Xavier and zeros initializers."""
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                           torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
        torch.nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


class ReshapeLayer(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self._shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, *self._shape)



def new_network(in_features: int, out_shape: tuple[int, ...], hidden_size: int) -> torch.nn.Sequential:
    out_features = np.prod(out_shape).item()  # 6 actions per player

    if hidden_size == 0:
        network = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            ReshapeLayer(out_shape)
        )
    else:
        network = torch.nn.Sequential(
            torch.nn.Linear(in_features, args.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, out_features),
            ReshapeLayer(out_shape)
        )
    return network

class Agent:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, args: argparse.Namespace, env: gym.Env) -> None:
        self._gamma = args.gamma
        self._lambd = args.lambd
        self._clip_eps = args.clip_eps
        self._entropy_reg = args.entropy_reg
        self._clip_grad_norm = args.clip_grad_norm
        self._tau = args.tau

        self._actor = new_network(
            env.observation_space.shape[0],  # type: ignore
            # (args.env_size * 2 + args.players) * (args.players + args.foods),
            (args.players, 6),
            args.hidden_size
        ).to(self.device)
        torch_init_with_orthogonal_and_zeros(self._actor)
        # with torch.no_grad():
            # self._actor[-2].weight.mul_(0.01)

        self._critic = new_network(
            env.observation_space.shape[0],  # type: ignore
            # 3 * (args.players + args.foods),
            (1,),
            args.hidden_size
        ).to(self.device)
        torch_init_with_orthogonal_and_zeros(self._critic)
        self._target_critic = copy.deepcopy(self._critic).to(self.device)

        self._params = [*self._actor.parameters(), *self._critic.parameters()]

        self._opt = torch.optim.Adam(self._params, args.lr)
        self._critic_loss = torch.nn.MSELoss()

    def predict_probs(self, states: np.ndarray) -> np.ndarray:
        t_states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        self._actor.eval()
        with torch.inference_mode():
            logits = self._actor(t_states)
        probs = torch.softmax(logits, -1)
        return probs.cpu().numpy()

    def predict_values(self, states: np.ndarray, target: bool = False) -> np.ndarray:
        t_states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        critic = self._target_critic if target else self._critic
        critic.eval()
        with torch.inference_mode():
            return critic(t_states).cpu().numpy()

    def _prepare_data(self, buffer: TrajectoryBuffer) -> tuple[np.ndarray, ...]:
        advantages, returns = compute_gae_and_ret(self, buffer, self._gamma, self._lambd)
        data = buffer.states, buffer.actions, buffer.action_probs, advantages, returns
        prev_dones = buffer.get_prev_dones()

        prep_data = tuple([] for _ in data)
        for t in range(len(data[0])):
            prev_not_dones = ~prev_dones[t]
            for prep_d, d in zip(prep_data, data):
                prep_d.extend(d[t][prev_not_dones])
        return tuple(map(np.array, prep_data))


    def train(self, buffer: TrajectoryBuffer, epochs: int, batch_size: int) -> None:
        for epoch in range(epochs):
            data = self._prepare_data(buffer)
            data_size = len(data[0])
            
            indices = np.random.permutation(data_size)
            for b in range(0, data_size - batch_size + 1, batch_size):
                batch_indices = indices[b : b + batch_size]
                self.train_step(*(d[batch_indices] for d in data))

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        actions_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> None:
        t_states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        t_actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        t_actions_probs = torch.as_tensor(actions_probs, dtype=torch.float32, device=self.device)
        t_advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        t_returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        self._actor.train()
        self._critic.train()

        logits = self._actor(t_states)
        probs = logits.softmax(-1)
        new_actions_probs = torch.take_along_dim(probs, t_actions[..., None], dim=-1).squeeze().prod(-1)
        rho = new_actions_probs / (t_actions_probs + 1e-8)
        # rho = probs[range(len(t_states)), t_actions] / (actions_probs + 1e-8)
        # rho = rho / rho.max()
        # rho = torch.clip(rho, 0, 1.5)
        ppo_loss = -torch.minimum(
            rho * t_advantages,
            torch.clip(rho, 1 - self._clip_eps, 1 + self._clip_eps) * t_advantages
        ).mean()

        entropy = torch.distributions.Categorical(logits=logits).entropy().mean()  # todo: ugly and wrong
        critic_loss = self._critic_loss(self._critic(t_states).squeeze(), t_returns)
        
        loss = critic_loss + ppo_loss - entropy * self._entropy_reg

        self._opt.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._params, self._clip_grad_norm)
        with torch.no_grad():
            self._opt.step()
        self.target_critic_update()

    def target_critic_update(self) -> None:
        with torch.no_grad():
            for source_param, target_param in zip(self._critic.parameters(), self._target_critic.parameters()):
                target_param.mul_(1 - self._tau)
                target_param.add_(source_param, alpha=self._tau)



def compute_gae_and_ret(
    agent: Agent,
    buffer: TrajectoryBuffer,
    gamma: float,
    lambd: float
) -> tuple[np.ndarray, np.ndarray]:
    gaes = np.empty_like(buffer.rewards)
    returns = np.empty_like(buffer.rewards)

    values_ext = agent.predict_values(buffer.extended_states(), True)  # todo: target jen pro kritika
    values = values_ext[:-1]
    v_last = values_ext[-1]

    gae = 0
    v_next = v_last
    for t in range(len(gaes) - 1, -1, -1):
        v = values[t]
        adv = buffer.rewards[t] + ~buffer.dones[t] * gamma * v_next - v
        gae = adv + gamma * lambd * gae

        gaes[t] = gae
        returns[t] = gae + v
    return gaes, returns



def evaluate_episode(agent: Agent, env: gym.Env, render: bool) -> float:
    def _render():
        if render:
            FPS = 5
            env.render()
            time.sleep(1/FPS)
            # input()

    def inverse_oh(state: np.ndarray) -> np.ndarray:
        state = state.reshape(args.players + args.foods, 2*args.env_size + args.players + 1)
        normal = np.empty([args.players + args.foods, 3])
        normal[:, 0] = state[:, :args.env_size].argmax(-1)
        normal[:, 1] = state[:, args.env_size:2*args.env_size].argmax(-1)
        normal[:, 2] = state[:, 2*args.env_size:].argmax(-1)
        return normal

    state = env.reset()[0]
    _render()

    done = False
    ret = 0
    while not done:
        probs = agent.predict_probs(state[None])[0]
        # print(probs)
        action = probs.argmax(-1)

        # print(inverse_oh(state))
        # env.render()
        # input()

        state, reward, terminated, truncated, _ = env.step(action)
        ret += float(reward)
        # print(probs, reward)
        # if reward > 0:
            # input()
        done = terminated or truncated

        _render()
        # print(action)
        # print(state.reshape(-1, 3))
        # input()
    return ret

def evaluate(agent: Agent, env: gym.Env, episodes: int, render_each: int) -> float:
    mean_ret = 0
    for episode in range(episodes):
        mean_ret += evaluate_episode(agent, env, episode % render_each == 0)
    mean_ret /= episodes
    return mean_ret

def main(args: argparse.Namespace) -> None:
    eval_env = gym.make("LBF")
    train_env = gym.vector.AsyncVectorEnv(
        [lambda: gym.make("LBF") for _ in range(args.num_envs)]
    )
    # train_env = NpWrapper(train_env)

    agent = Agent(args, eval_env)
    buffer = TrajectoryBuffer()

    states = train_env.reset()[0]

    prev_dones = np.zeros(train_env.num_envs, dtype=bool)

    for train_step in range(1, args.train_steps + 1):
        buffer.clear()
        for update_step in range(1, args.steps_per_update + 1):
            probs = agent.predict_probs(states)
            actions = torch.distributions.Categorical(torch.as_tensor(probs)).sample().numpy()

            next_states, rewards, terminations, truncations, _ = train_env.step(actions)

            dones = terminations | truncations
            values = agent.predict_values(states)
            actions_probs = np.take_along_axis(probs, actions[..., None], axis=-1).squeeze().prod(-1)

            buffer.append(states, actions, actions_probs, rewards, dones, values, next_states, prev_dones)

            states = next_states
            prev_dones = dones

        # print(train_step)
        agent.train(buffer, args.epochs, args.batch_size)


        if train_step % args.eval_each == 0:
            print(f"Evaluation after {train_step} steps: ", end="", flush=True)
            mean_return = evaluate(agent, eval_env, args.eval_for, args.render_each)
            print(f"{mean_return:.3f}")

class NpWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env):
        gym.utils.RecordConstructorArgs.__init__(self)
        super().__init__(env)
        # self._observation_space = gym.spaces.MultiBinary(
            # [(args.players + args.foods) * (args.env_size * 2 + args.players + 1)]
        # )
        self._observation_space = gym.spaces.Box(
            np.zeros((args.players + args.foods) * 3, dtype=np.float32),
            np.array([args.env_size, args.env_size, args.players] * (args.players + args.foods), dtype=np.float32),
        )
        self._action_space = gym.spaces.MultiDiscrete([6]*args.players)

    def _full_observation_tripples(self, obs: tuple[np.ndarray, ...]) -> np.ndarray:
        """Original observation is a tuple of flattened tripples (row, col, level):
        food 1, food 2, ..., food F, agent i, other agents' positions...

        Returns: ndarray (foods + players, 3)

        Already eaten food is reprezented by zeros.
        We support only fully observable version of the environment.
        """
        foods = obs[0].astype(np.int64).reshape(-1, 3)[:args.foods]
        padded_foods = np.zeros([args.foods, 3], dtype=np.int64)
        padded_foods[:len(foods)] = foods
        positions = np.array([ob.astype(np.int64).reshape(-1, 3)[args.foods] for ob in obs])
        new_obs = np.concat([padded_foods, positions], axis=0)
        return new_obs

    def _three_hot_observation(self, obs: np.ndarray) -> np.ndarray:
        """Makes three-hot reprezentation.
        Transforms each value in the tripplet into one-hot encoding.

        Returns: ndarray, ((foods + players), (env_size * 2 + players))
        """
        new_obs = np.zeros(((args.foods + args.players), (args.env_size * 2 + args.players + 1)), dtype=np.int64)
        _rang = np.arange(args.foods + args.players)
        new_obs[_rang, obs[:, 0]] = 1
        new_obs[_rang, args.env_size + obs[:, 1]] = 1
        new_obs[_rang, 2*args.env_size + obs[:, 2]] = 1
        return new_obs

    def observation(self, obs):
        new_obs = self._full_observation_tripples(obs)

        new_obs = new_obs.astype(np.float64)
        new_obs[:, :2] /= args.env_size
        new_obs[:, 2] /= args.players

        # new_obs = self._three_hot_observation(new_obs)
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
        


# class NpWrapper(gym.vector.VectorWrapper):
    # """Vectorization returns list of ndarrays for some reason"""
    # def __init__(self, env):
        # super().__init__(env)

    # def reset(self, *, seed=None, options=None):
        # new_obs, info = super().reset(seed=seed, options=options)
        # return np.asarray(new_obs), info

    # def step(self, actions):
        # obs, rewards, terminations, truncations, info = super().step(actions)
        # obs = np.asarray(obs)

        # # Since we will do cooperation only let's average.
        # # I have no interest in inspecting what the actual rewards are.
        # rewards = np.asarray(rewards).mean(-1)

        # terminations = np.asarray(terminations)
        # truncations = np.asarray(truncations)

        # return obs, rewards, terminations, truncations, info


# python main.py --env_size 8 --players 2 --foods 3 --steps_per_update 10 --batch_size 10 --tau 0.01 --lr 3e-4 --epochs 4 --hidden_size 128 --entropy_reg 0.001 --clip_eps 0.2 --num_envs 10 --eval_each 1000 --eval_for 10
if __name__ == "__main__":
    args = parser.parse_args()
    gym.register(
        id="LBF",
        entry_point="lbforaging.foraging:ForagingEnv",
        disable_env_checker=True,
        additional_wrappers=(NpWrapper.wrapper_spec(),),
        kwargs={
            "players": args.players,
            "min_player_level": np.ones(args.players),
            "max_player_level": np.full(args.players, args.players),
            "field_size": (args.env_size, args.env_size),
            "min_food_level": np.ones(args.foods),
            "max_food_level": np.arange(1, args.foods + 1).clip(1, args.players),
            # "max_food_level": None,#np.arange(1, args.foods + 1).clip(1, args.players),
            "max_num_food": args.foods,
            "sight": args.env_size,  # see the whole board
            "max_episode_steps": args.ep_limit,
            "force_coop": True,
            "grid_observation": False,
            "penalty": 0.,
        },
    )
    # players,
    # min_player_level,
    # max_player_level,
    # min_food_level,
    # max_food_level,
    # field_size,
    # max_num_food,
    # sight,
    # max_episode_steps,
    # force_coop,
    # normalize_reward=True,
    # grid_observation=False,
    # observe_agent_levels=True,
    # penalty=0.0,
    # render_mode=None,
    main(args)
