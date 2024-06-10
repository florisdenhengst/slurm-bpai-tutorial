import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from pettingzoo.atari import entombed_cooperative_v3
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""
    env_id: str = "entombed_cooperative_v3"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10


def make_env(seed, capture_video, run_name):
    def thunk():
        env = entombed_cooperative_v3.parallel_env(max_cycles=500)
        env = color_reduction_v0(env)
        env = resize_v1(env, 64, 64)
        env = frame_stack_v1(env, 4)
        if capture_video:
            from gym.wrappers import Monitor
            env = Monitor(env, f'videos/{run_name}')
        return env

    return thunk


class QNetwork(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def select_action(obs, q_network, epsilon, device, action_space, possible_agents):
    actions = {}
    for agent in possible_agents:
        if random.random() < epsilon:
            actions[agent] = action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs[agent]).unsqueeze(0).to(device))
            actions[agent] = torch.argmax(q_values, dim=1).cpu().numpy()[0]
    return actions


def optimize_model(rb, q_network, target_network, optimizer, batch_size, gamma, device):
    data = rb.sample(batch_size)
    with torch.no_grad():
        target_max, _ = target_network(data.next_observations).max(dim=1)
        td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())
    old_val = q_network(data.observations).gather(1, data.actions).squeeze()
    loss = F.mse_loss(td_target, old_val)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, old_val


def log_metrics(writer, global_step, loss, old_val, start_time):
    writer.add_scalar("losses/td_loss", loss, global_step)
    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
    sps = int(global_step / (time.time() - start_time))
    writer.add_scalar("charts/SPS", sps, global_step)
    print("SPS:", sps)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_env(args.seed, args.capture_video, run_name)()
    env.reset(seed=args.seed)
    action_space = env.action_space(env.possible_agents[0])
    q_network = QNetwork(action_space.n).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space(env.possible_agents[0]),
        env.action_space(env.possible_agents[0]),
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    obs = env.reset(seed=args.seed)
    
    # Debug print statements
    print(f"Initial observation: {obs}")

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        actions = select_action(obs, q_network, epsilon, device, action_space, env.possible_agents)
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # Debug print statements to understand the structure
        print(f"Step {global_step}")
        print(f"Observations: {obs}")
        print(f"Next Observations: {next_obs}")
        print(f"Rewards: {rewards}")
        print(f"Terminations: {terminations}")
        print(f"Truncations: {truncations}")
        print(f"Infos: {infos}")

        for agent in env.possible_agents:
            real_next_obs = next_obs[agent] if isinstance(next_obs, dict) else next_obs[env.possible_agents.index(agent)]
            if truncations[agent] if isinstance(truncations, dict) else truncations[env.possible_agents.index(agent)]:
                real_next_obs = infos[agent]["final_observation"] if isinstance(infos, dict) else infos[env.possible_agents.index(agent)]["final_observation"]
            rb.add(
                obs[agent] if isinstance(obs, dict) else obs[env.possible_agents.index(agent)],
                real_next_obs,
                actions[agent] if isinstance(actions, dict) else actions[env.possible_agents.index(agent)],
                rewards[agent] if isinstance(rewards, dict) else rewards[env.possible_agents.index(agent)],
                terminations[agent] if isinstance(terminations, dict) else terminations[env.possible_agents.index(agent)],
                infos[agent] if isinstance(infos, dict) else infos[env.possible_agents.index(agent)]
            )

        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                loss, old_val = optimize_model(rb, q_network, target_network, optimizer, args.batch_size, args.gamma, device)
                if global_step % 100 == 0:
                    log_metrics(writer, global_step, loss, old_val, start_time)

            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    env.close()
    writer.close()
