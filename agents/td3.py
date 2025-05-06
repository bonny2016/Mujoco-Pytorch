import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from networks.network import Actor, Critic
from utils.utils import ReplayBuffer, convert_to_tensor

class TD3(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, args, noise, action_max):
        super(TD3, self).__init__()
        self.device = device
        self.writer = writer
        self.args = args
        self.action_max = action_max
        self.noise = noise
        self.train_step = 0

        # Networks
        self.actor = Actor(args.layer_num, state_dim, action_dim, args.hidden_dim,
                           args.activation_function, args.last_activation, args.trainable_std, use_layernorm=args.use_layernorm)
        self.target_actor = Actor(args.layer_num, state_dim, action_dim, args.hidden_dim,
                                  args.activation_function, args.last_activation, args.trainable_std, use_layernorm=args.use_layernorm)

        self.q1 = Critic(args.layer_num, state_dim + action_dim, 1, args.hidden_dim, args.activation_function, None)
        self.q2 = Critic(args.layer_num, state_dim + action_dim, 1, args.hidden_dim, args.activation_function, None)
        self.target_q1 = Critic(args.layer_num, state_dim + action_dim, 1, args.hidden_dim, args.activation_function, None)
        self.target_q2 = Critic(args.layer_num, state_dim + action_dim, 1, args.hidden_dim, args.activation_function, None)

        self.soft_update(self.q1, self.target_q1, 1.)
        self.soft_update(self.q2, self.target_q2, 1.)
        self.soft_update(self.actor, self.target_actor, 1.)

        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=args.q_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        # Replay buffer
        self.data = ReplayBuffer(action_prob_exist=False, max_size=int(args.memory_size),
                                 state_dim=state_dim, num_action=action_dim)

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_action(self, x):
        mu, std = self.actor(x)
        action = mu + torch.tensor(self.noise.sample(), device=self.device, dtype=mu.dtype)
        return (action * self.action_max).clamp(-self.action_max, self.action_max), std

    def get_deterministic_action(self, x):
        mu, _ = self.actor(x)
        return (mu * self.action_max).clamp(-self.action_max, self.action_max)

    def put_data(self, transition):
        self.data.put_data(transition)

    def train_net(self, batch_size, n_epi):
        data = self.data.sample(shuffle=True, batch_size=batch_size)
        states, actions, rewards, next_states, dones = convert_to_tensor(
            self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done']
        )

        with torch.no_grad():
            next_action, _ = self.target_actor(next_states)
            noise = (torch.randn_like(next_action) * self.args.policy_noise).clamp(-self.args.noise_clip, self.args.noise_clip)
            noisy_next_action = (next_action + noise).clamp(-1, 1) * self.action_max
            # noisy_next_action = (next_action).clamp(-1, 1) * self.action_max
            target_q1_val = self.target_q1(next_states, noisy_next_action)
            target_q2_val = self.target_q2(next_states, noisy_next_action)
            target_q = torch.min(target_q1_val, target_q2_val)
            targets = rewards + self.args.gamma * (1 - dones) * target_q

        # Critic 1
        q1_loss = F.mse_loss(self.q1(states, actions), targets)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=1.0)
        self.q1_optimizer.step()

        # Critic 2
        q2_loss = F.mse_loss(self.q2(states, actions), targets)
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=1.0)
        self.q2_optimizer.step()

        # Delayed policy update
        if self.train_step % self.args.policy_delay == 0:
            actor_loss = -self.q1(states, self.actor(states)[0]).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            self.soft_update(self.q1, self.target_q1, self.args.soft_update_rate)
            self.soft_update(self.q2, self.target_q2, self.args.soft_update_rate)
            self.soft_update(self.actor, self.target_actor, self.args.soft_update_rate)

            if self.writer:
                self.writer.add_scalar("loss/actor", actor_loss, n_epi)

        if self.writer:
            self.writer.add_scalar("loss/q1", q1_loss, n_epi)
            self.writer.add_scalar("loss/q2", q2_loss, n_epi)

        self.train_step += 1
