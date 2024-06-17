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
from pettingzoo.atari import entombed_competitive_v3
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
        #removed max cycles
        env = entombed_cooperative_v3.parallel_env()
        env = color_reduction_v0(env)
        env = resize_v1(env, 84, 84)
        env = frame_stack_v1(env, 4)
        #if args.capture_video:
            #env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
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
            #flatten does 64x64 and then / 4 for the frame stack, so it becomes 1024?
            nn.Flatten(),
            #if i change this so the dimensions are the same, it does not run at all(no print statemetns so it does not start the looop)
            #nn.Linear(1024, 512),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
#

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
    # added variables for everything for player 2
    action_space = env.action_space(env.possible_agents[0]).n
    action_space2 = env.action_space(env.possible_agents[1]).n

    q_network = QNetwork(action_space).to(device)
    q_network2 = QNetwork(action_space2).to(device)

    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    optimizer2 = optim.Adam(q_network2.parameters(), lr=args.learning_rate)

    target_network = QNetwork(action_space).to(device)
    target_network2 = QNetwork(action_space2).to(device)

    target_network.load_state_dict(q_network.state_dict())
    target_network2.load_state_dict(q_network2.state_dict())


    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space(env.possible_agents[0]),
        env.action_space(env.possible_agents[0]),
        device,
        handle_timeout_termination=False,
    )
    # added replay buffer for p2
    rb2 = ReplayBuffer(
        args.buffer_size,
        env.observation_space(env.possible_agents[1]),
        env.action_space(env.possible_agents[1]),
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    obs, _ = env.reset(seed=args.seed)
    q_values_dict = {}  # Dictionary to store Q-values

    print('start main')
    reward_lst = {'first_0': 0, 'second_0': 0}
    current_time = 0
    for global_step in range(args.total_timesteps):
        if global_step % 100 == 0:
            print(f"Global step: {global_step}")
        # Log Q-values at the start of each episode
        # if global_step % 1000 == 0:
        #     q_values = q_network(torch.Tensor(obs).to(device))
        #     writer.add_histogram("q_values", q_values, global_step)
        #     print(f"Q-values at step {global_step}: {q_values.cpu().detach().numpy()}")
            
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        actions = {}
        actions2 = {}
        for agent in env.possible_agents:
            if agent == 'first_0':
                #print(agent)
                #print(env.possible_agents)
                if random.random() < epsilon:
                    actions[agent] = env.action_space(agent).sample()
                else:
                    
                    q_values = q_network(torch.Tensor(obs[agent]).permute((2,0,1)).unsqueeze(0).to(device))
                    actions[agent] = torch.argmax(q_values, dim=1).cpu().numpy()[0]
            elif agent == 'second_0':
                #print(agent)
                #print(env.possible_agents)
                if random.random() < epsilon:
                    actions[agent] = env.action_space(agent).sample()
                else:
                   
                    #print((torch.Tensor(obs[agent]).permute((2,0,1)).unsqueeze(0).to(device)).shape)
                    q_values2 = q_network2(torch.Tensor(obs[agent]).permute((2,0,1)).unsqueeze(0).to(device))
                    actions[agent] = torch.argmax(q_values2, dim=1).cpu().numpy()[0]
        
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # reset manually
        
        if not(env.agents):
            writer.add_scalar("charts/episodic_return", reward_lst["first_0"], global_step)
            writer.add_scalar("charts/episodic_length", global_step-current_time, global_step)
            print("All agents done, resetting environment.")
            obs, _ = env.reset(seed=args.seed)
            reward_lst = {'first_0': 0, 'second_0': 0}
            current_time = global_step

        else:
            for agent in env.possible_agents:
                # might need to also add the agent['first_0'] to all the agent but im not sure
                if agent == 'first_0':
                    real_next_obs = next_obs[agent]
                    if truncations[agent]:
                        real_next_obs = infos[agent]["final_observation"]
                    reward_lst['first_0']+= rewards[agent]
                    if 'q_values' in locals():
                        q_values_dict[f"{global_step}_{agent}"] = q_values.cpu().detach().numpy().tolist()
                    rb.add(obs[agent], real_next_obs, actions[agent], rewards[agent], terminations[agent], infos[agent])

                elif agent == 'second_0':
                    real_next_obs = next_obs[agent]
                    if truncations[agent]:
                        real_next_obs = infos[agent]["final_observation"]
                    reward_lst['second_0']+= rewards[agent]
                    rb2.add(obs[agent], real_next_obs, actions[agent], rewards[agent], terminations[agent], infos[agent])

        
# now im here
        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                print("Optimizing models...")
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations.permute(0, 3, 1, 2)).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations.permute(0, 3, 1, 2)).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)
                # p2
                data2 = rb2.sample(args.batch_size)
                with torch.no_grad():
                    target_max2, _ = target_network2(data.next_observations.permute(0, 3, 1, 2)).max(dim=1)
                    td_target2 = data2.rewards.flatten() + args.gamma * target_max2 * (1 - data2.dones.flatten())
                old_val2 = q_network2(data2.observations.permute(0, 3, 1, 2)).gather(1, data2.actions).squeeze()
                loss2 = F.mse_loss(td_target2, old_val2)

                if global_step % 100 == 0:
                    #writer.add_histogram("q_values", q_values, global_step)
                    print(f"Q-values at step {global_step}: {q_values.cpu().detach().numpy()}")

                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    #p2
                    writer.add_scalar("losses/td_loss2", loss2, global_step)
                    writer.add_scalar("losses/q_values2", old_val2.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    #writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #p2
                optimizer2.zero_grad()
                loss2.backward()
                optimizer2.step()

            if global_step % args.target_network_frequency == 0:
                print("Updating target networks...")
                target_network.load_state_dict(q_network.state_dict())
                target_network2.load_state_dict(q_network2.state_dict())
    print("Training completed.")
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        torch.save(q_network2.state_dict(), model_path + "_p2")
        print(f"model saved to {model_path}")

        

    env.close()
    writer.close()