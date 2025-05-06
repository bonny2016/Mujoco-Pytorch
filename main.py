import os
import torch
import numpy as np
import gym
from configparser import ConfigParser
from argparse import ArgumentParser

from agents.ppo import PPO
from agents.sac import SAC
from agents.ddpg import DDPG
from agents.td3 import TD3
from utils.noise import OUNoise
from utils.utils import make_transition, Dict, RunningMeanStd

def evaluate(agent, env, device, episodes=5):
    total_return = 0.0
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            state_tensor = torch.from_numpy(state).float().to(device)
            with torch.no_grad():
                action = agent.get_deterministic_action(state_tensor)
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            ep_return += reward
            state = next_state
        total_return += ep_return
        print(f"[Eval] Episode {ep+1} return: {ep_return:.2f}")
    avg_return = total_return / episodes
    print(f"[Evaluation only] Average return over {episodes} episodes: {avg_return:.2f}")
    return avg_return

def main():
    parser = ArgumentParser('parameters')
    parser.add_argument("--env_name", type=str, default='Hopper-v4',
                        help = "'Ant-v2','HalfCheetah-v4','Hopper-v4','Humanoid-v4','HumanoidStandup-v4',\
                                'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v4)")
    parser.add_argument("--algo", type=str, default='td3')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--tensorboard', type=bool, default=False)
    parser.add_argument("--load", type=str, default='no')
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--print_interval", type=int, default=1)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--reward_scaling", type=float, default=1)
    args = parser.parse_args()
    os.makedirs(f'./model_weights/{args.algo}/{args.env_name}', exist_ok=True)
    config = ConfigParser()
    config.read('config.ini')
    agent_args = Dict(config, args.algo)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not args.use_cuda:
        device = 'cpu'

    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join("runs", f"{args.env_name}_{args.algo}")
        writer = SummaryWriter(log_dir=log_dir)
    env = gym.make(args.env_name, render_mode="human" if args.render else None)
    action_dim = env.action_space.shape[0]
    action_max = float(env.action_space.high[0])
    state_dim = env.observation_space.shape[0]
    state_rms = RunningMeanStd(state_dim)

    if args.algo == 'ppo':
        agent = PPO(writer, device, state_dim, action_dim, agent_args)
    elif args.algo == 'sac':
        agent = SAC(writer, device, state_dim, action_dim, agent_args)
    elif args.algo == 'ddpg':
        noise = OUNoise(mu=np.zeros(action_dim))
        agent = DDPG(writer, device, state_dim, action_dim, agent_args, noise, action_max=action_max)
    elif args.algo == 'td3':
        noise = OUNoise(mu=np.zeros(action_dim))
        agent = TD3(writer, device, state_dim, action_dim, agent_args, noise, action_max=action_max)

    if torch.cuda.is_available() and args.use_cuda:
        agent = agent.cuda()

    if args.load != 'no':
        device_map = torch.device("cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu")
        agent.load_state_dict(torch.load(f'./model_weights/{args.algo}/{args.env_name}/{args.load}', map_location=device_map))
        # agent.load_state_dict(torch.load(f'./model_weights/{args.algo}/{args.env_name}/' + args.load))

    if not args.train:
        evaluate(agent, env, device)
        return

    score_lst = []
    state_lst = []
    best_score = -np.inf

    if agent_args.on_policy:
        score = 0.0
        state_, _ = env.reset()
        state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
        for n_epi in range(args.epochs):
            for t in range(agent_args.traj_length):
                if args.render:
                    env.render()
                state_lst.append(state_)
                mu, sigma = agent.get_action(torch.from_numpy(state).float().to(device))
                dist = torch.distributions.Normal(mu, sigma[0])
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1, keepdim=True)
                next_state_, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                done = terminated or truncated
                next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                transition = make_transition(state,
                                             action.cpu().numpy(),
                                             np.array([reward * args.reward_scaling]),
                                             next_state,
                                             np.array([done]),
                                             log_prob.detach().cpu().numpy())
                agent.put_data(transition)
                score += reward
                if done:
                    state_, _ = env.reset()
                    state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                    score_lst.append(score)
                    if args.tensorboard:
                        writer.add_scalar("score/score", score, n_epi)
                    score = 0
                else:
                    state = next_state
                    state_ = next_state_

            agent.train_net(n_epi)
            state_rms.update(np.vstack(state_lst))
            if n_epi % args.print_interval == 0 and n_epi != 0:
                print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
                score_lst = []
            if n_epi % args.save_interval == 0 and n_epi != 0:
                torch.save(agent.state_dict(), f'./model_weights/{args.algo}/{args.env_name}/agent_' + str(n_epi))
                avg_score = sum(score_lst) / len(score_lst)
                if avg_score > best_score:
                    best_score = avg_score
                    torch.save(agent.state_dict(), f'./model_weights/{args.algo}/{args.env_name}/best_agent.pth')
                    print(f"[INFO] New best model saved with avg score: {avg_score:.2f}")


    else:
        for n_epi in range(args.epochs):
            score = 0.0
            state, _ = env.reset()
            done = False
            while not done:
                if args.render:
                    env.render()
                action, _ = agent.get_action(torch.from_numpy(state).float().to(device))
                action = action.cpu().detach().numpy()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                transition = make_transition(state,
                                             action,
                                             np.array([reward * args.reward_scaling]),
                                             next_state,
                                             np.array([done]))
                agent.put_data(transition)
                state = next_state
                score += reward
                if hasattr(agent, 'data') and agent.data.data_idx > agent_args.learn_start_size:
                    agent.train_net(agent_args.batch_size, n_epi)

            score_lst.append(score)
            if args.tensorboard:
                writer.add_scalar("score/score", score, n_epi)
            if n_epi % args.print_interval == 0 and n_epi != 0:
                print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
                score_lst = []
            if n_epi % args.save_interval == 0 and n_epi != 0:
                torch.save(agent.state_dict(),  f'./model_weights/{args.algo}/{args.env_name}/agent_' + str(n_epi))
            if args.algo == 'ddpg':
                agent.noise.decay_sigma()
            if args.algo == 'td3':
                agent.noise.decay_sigma()

if __name__ == "__main__":
    main()
