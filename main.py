import numpy as np
import torch
import gymnasium as gym

import argparse
import time

import wrappers

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="model", help="path to the model file")
parser.add_argument("--hidden_size", type=int, default=0, help="size of hidden layer")
parser.add_argument("--num_envs", type=int, default=8, help="number of parallel training envs")
parser.add_argument("--ep_limit", type=int, default=3, help="number of parallel training envs")
parser.add_argument("--eval_each", type=int, default=1000, help="evaluation period")
parser.add_argument("--eval_for", type=int, default=100, help="number of eval_episodes")
parser.add_argument("--render_each", type=int, default=50, help="render eval eps")

parser.add_argument("--env_size", type=int, default=3, help="size of the envirnonment")
parser.add_argument("--players", type=int, default=1, help="number of players")
parser.add_argument("--foods", type=int, default=1, help="number of players")

parser.add_argument("--train_steps", type=int, default=1_000_000, help="number of training steps")
parser.add_argument("--steps_per_update", type=int, default=10, help="number of steps in the env per update")
parser.add_argument("--batch_size", type=int, default=16, help="size of a single batch")
parser.add_argument("--epochs", type=int, default=2, help="number of epochs per training")
parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="number of steps in the env per update")
parser.add_argument("--lambd", type=float, default=0.95, help="gae trace lambda")
parser.add_argument("--clip_eps", type=float, default=0.25, help="ppo clip")
parser.add_argument("--entropy_reg", type=float, default=0.001, help="entropy regularization")
parser.add_argument("--clip_grad_norm", type=float, default=10., help="gradient clipping")

parser.add_argument("--three_hot", action="store_true", default=False, help="uses three-hot states")
parser.add_argument("--reward_shaping", action="store_true", default=False, help="uses reward shaping")
parser.add_argument("--live_plot", action="store_true", default=False, help="plots evaluation performance")
parser.add_argument("--evaluate", action="store_true", default=False, help="evaluates selected model")



class TrajectoryBuffer:
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        # self.values = []
        self.last_next_states = None
        self.prev_dones = None
        
    def append(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        actions_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        # values: np.ndarray,
        next_states: np.ndarray,
        prev_dones: np.ndarray
    ) -> None:
        self.states.append(states)
        self.actions.append(actions)
        self.action_probs.append(actions_probs)
        self.rewards.append(rewards)
        self.dones.append(dones)
        # self.values.append(values)
        self.last_next_states = next_states
        if self.prev_dones is None:
            self.prev_dones = prev_dones

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.action_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        # self.values.clear()
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
    """Initialize weights of a PyTorch module with orthogonal and zeros initializers."""
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                           torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
        torch.nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


class ReshapeLayer(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self._shape = (-1, *shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self._shape)



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

        self._actor = new_network(
            env.observation_space.shape[0],  # type: ignore
            # (args.env_size * 2 + args.players) * (args.players + args.foods),
            (args.players, 6),
            args.hidden_size
        ).to(self.device)
        self._actor.apply(torch_init_with_orthogonal_and_zeros)
        with torch.no_grad():
            self._actor[-2].weight.mul_(0.01)

        self._critic = new_network(
            env.observation_space.shape[0],  # type: ignore
            # 3 * (args.players + args.foods),
            (1,),
            args.hidden_size
        ).to(self.device)
        self._critic.apply(torch_init_with_orthogonal_and_zeros)

        self._params = [*self._actor.parameters(), *self._critic.parameters()]

        self._opt = torch.optim.Adam(self._params, args.lr, eps=1e-5)
        self._critic_loss = torch.nn.MSELoss()

    def predict_probs(self, states: np.ndarray) -> np.ndarray:
        t_states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        self._actor.eval()
        with torch.inference_mode():
            logits = self._actor(t_states)
        probs = torch.softmax(logits, -1)
        return probs.cpu().numpy()

    def predict_values(self, states: np.ndarray) -> np.ndarray:
        t_states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        self._critic.eval()
        with torch.inference_mode():
            return self._critic(t_states).cpu().numpy()

    def _prepare_data(self, buffer: TrajectoryBuffer) -> tuple[np.ndarray, ...]:
        advantages, returns = compute_gae_and_ret(self, buffer, self._gamma, self._lambd)
        data = buffer.states, buffer.actions, buffer.action_probs, advantages, returns
        prev_dones = buffer.get_prev_dones()

        # print("trajectory states", np.array(data[0]).shape)
        # print(prev_dones.shape, "valid transitions:", np.count_nonzero(~prev_dones))
        # print("total rew:", np.sum(buffer.rewards))

        # print()
        # print("rewards", np.array(buffer.rewards)[:, 0])
        # print("returns", returns[:, 0])
        # print("advantages", advantages[:, 0])
        # print("e dones", buffer.extended_dones()[:, 0])
        # print("p dones", prev_dones[:, 0])
        # exit()

        prep_data = tuple([] for _ in data)
        for t in range(len(data[0])):
            prev_not_dones = ~prev_dones[t]
            for prep_d, d in zip(prep_data, data):
                prep_d.extend(d[t][prev_not_dones])

        # print(np.array(data[0]).shape)
        # print(np.array(prep_data[-1]))
        # exit()
        return tuple(map(np.array, prep_data))


    def train(self, buffer: TrajectoryBuffer, epochs: int, batch_size: int) -> None:
        for epoch in range(epochs):
            data = self._prepare_data(buffer)
            data_size = len(data[0])
            # for d in data:
                # print(d.shape)
            # exit()
            
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

        # t_advantages = (t_advantages - t_advantages.mean()) / (t_advantages.std() + 1e-8)

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
        torch.nn.utils.clip_grad_norm_(self._params, self._clip_grad_norm)
        with torch.no_grad():
            self._opt.step()

    def save(self, path: str) -> None:
        torch.save(self._actor.state_dict(), path)

    def load(self, path: str) -> None:
        self._actor.load_state_dict(torch.load(path, map_location=Agent.device))



def compute_gae_and_ret(
    agent: Agent,
    buffer: TrajectoryBuffer,
    gamma: float,
    lambd: float
) -> tuple[np.ndarray, np.ndarray]:
    gaes = np.empty_like(buffer.rewards)
    returns = np.empty_like(buffer.rewards)

    ext_states = buffer.extended_states()
    T, E, S = ext_states.shape

    values_ext = agent.predict_values(ext_states.reshape(-1, S)).reshape(T, E)
    values = values_ext[:-1]
    v_last = values_ext[-1]
    # print("exs", buffer.extended_states().shape)
    # print(values_ext.shape, values.shape, v_last.shape)
    # exit()

    gae = 0
    v_next = v_last
    for t in range(len(gaes) - 1, -1, -1):
        v = values[t]
        # print("v", v.shape)

        adv = buffer.rewards[t] + ~buffer.dones[t] * gamma * v_next - v
        # print(f"[{t}] adv = {adv[0]} = {buffer.rewards[t][0]} + {~buffer.dones[t][0]} * {gamma} * {v_next[0]} - {v[0]}")
        gae = adv + ~buffer.dones[t] * gamma * lambd * gae

        gaes[t] = gae
        returns[t] = gae + v
        # print(returns[t].shape, "=", gae.shape, "+", v.shape)
        # print(f"[{t}] ret = {(gae + v)}")

        # if buffer.dones[t][0] and buffer.rewards[t][0] == 0 and t < args.ep_limit - 1:
            # print(buffer.states[t][0].reshape(2, -1))
            # print(buffer.states[t + 1][0].reshape(2, -1))

        v_next = v
    # exit()
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

    # steps = 0
    # food = state

    while not done:
        # steps += 1
        # print(state.shape)
        probs = agent.predict_probs(state)[0]
        action = probs.argmax(-1)
        # print(probs.shape)
        # print(action.shape)
        # exit()

        # row = state.reshape(2, -1)[1][:args.env_size].argmax()
        # col = state.reshape(2, -1)[1][args.env_size : args.env_size * 2].argmax()
        # up = row == 0
        # down = row == args.env_size - 1
        # left = col == 0
        # right = col == args.env_size - 1
        # in_corner = (up or down) and (left or right)

        # if in_corner:
            # a = 2 if up else 1
        # else:
            # a = 5
        # action =  np.full(args.players, a)

        state, reward, terminated, truncated, _ = env.step(action)
        ret += float(reward)
        # print(probs, reward)
        done = terminated or truncated

        _render()
        # print(action)
        # print(state.reshape(-1, 3))
        # input()
    # print(f"steps: {steps}, return: {ret}, {food}")
    return ret

def evaluate(agent: Agent, env: gym.Env, episodes: int, render_each: int) -> float:
    mean_ret = 0
    for episode in range(episodes):
        mean_ret += evaluate_episode(agent, env, episode % render_each == 0)
    mean_ret /= episodes
    return mean_ret

def new_env_with_wrappers(args: argparse.Namespace, eval: bool = False) -> gym.Env:
    """gym.make might have something better than this function"""
    env = gym.make("LBF")
    env = wrappers.NpWrapper(env, args)
    if args.reward_shaping and not eval:
        assert not args.three_hot, "Reward shaping does not support 3-hot states!"
        env = wrappers.RewardShapingWrapper(env, args)
    if args.live_plot and eval:
        env = wrappers.LivePlotWrapper(env, args.eval_for)
    return env

def main(args: argparse.Namespace) -> None:
    eval_env = new_env_with_wrappers(args, True)
    agent = Agent(args, eval_env)

    if args.evaluate:
        agent.load(args.model)
        while True:
            mean_return = evaluate(agent, eval_env, args.eval_for, args.render_each)
            print(f"Mean {args.eval_for} return: {mean_return:.3f}")
        return

    train_env = gym.vector.AsyncVectorEnv(
        [lambda: new_env_with_wrappers(args) for _ in range(args.num_envs)]
    )


    buffer = TrajectoryBuffer()
    states = train_env.reset()[0]

    prev_dones = np.zeros(train_env.num_envs, dtype=bool)

    for train_step in range(1, args.train_steps + 1):
        buffer.clear()
        for update_step in range(1, args.steps_per_update + 1):

            # def fn(name, item):
                # print(name)
                # print("    pos:", item[:args.env_size].argmax().item(), item[args.env_size:2*args.env_size].argmax().item())
                # print("    lvl:", item[2*args.env_size:].argmax().item())

            # ss = states[0].reshape(args.foods+args.players, -1)
            # fs = ss[:args.foods]
            # ps = ss[args.foods:]
            # for i in range(args.foods):
                # fn(f"food {i}", fs[i])
            # for i in range(args.players):
                # fn(f"player {i}", ps[i])
            # train_env.envs[0].render()
            # input()
            # exit()

            probs = agent.predict_probs(states)
            # print(probs[0])
            actions = torch.distributions.Categorical(torch.as_tensor(probs)).sample().numpy()

            next_states, rewards, terminations, truncations, _ = train_env.step(actions)

            dones = terminations | truncations
            # values = agent.predict_values(states)
            actions_probs = np.take_along_axis(probs, actions[..., None], axis=-1).squeeze(-1).prod(-1)

            buffer.append(states, actions, actions_probs, rewards, dones, next_states, prev_dones)



            states = next_states
            prev_dones = dones

        # print(train_step)
        agent.train(buffer, args.epochs, args.batch_size)


        if train_step % args.eval_each == 0:
            print(f"Evaluation after {train_step} steps: ", end="", flush=True)
            mean_return = evaluate(agent, eval_env, args.eval_for, args.render_each)
            print(f"{mean_return:.3f}")
    agent.save(args.model)



# python main.py --entropy_reg 0.01 --lr 0.001 --players 2 --env_size 7 --foods 3 --hidden_size 128 --ep_limit 32 --steps_per_update 16 --batch_size 32

if __name__ == "__main__":
    args = parser.parse_args()
    gym.register(
        id="LBF",
        entry_point="lbforaging.foraging:ForagingEnv",
        disable_env_checker=True,
        # additional_wrappers=(
            # wrappers.NpWrapper.wrapper_spec(args=args),
            # *([wrappers.RewardShapingWrapper.wrapper_spec(args=args)] if args.reward_shaping else []),
        # ),
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
