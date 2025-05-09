from networks.network import Actor, Critic
from utils.utils import ReplayBuffer, convert_to_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal


class DDPG(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, args, noise, action_max):
        super(DDPG,self).__init__()
        self.device = device
        self.writer = writer
        self.action_max = action_max
        self.args = args
        self.actor = Actor(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, \
                           self.args.activation_function, self.args.last_activation, self.args.trainable_std, use_layernorm=True)
        
        self.target_actor = Actor(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, \
                           self.args.activation_function, self.args.last_activation, self.args.trainable_std,use_layernorm=True)
        
        self.q = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function, None, use_layernorm=True)
        
        self.target_q = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function, None, use_layernorm=True)
        
        self.soft_update(self.q, self.target_q, 1.)
        self.soft_update(self.actor, self.target_actor, 1.)
        
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=self.args.q_lr)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.data = ReplayBuffer(action_prob_exist = False, max_size = int(self.args.memory_size), state_dim = state_dim, num_action = action_dim)
        
        self.noise = noise
        
    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)

    def get_action(self, x):
        mu, std = self.actor(x)
        action = mu + torch.tensor(self.noise.sample(), device=self.device, dtype=mu.dtype)
        return (action * self.action_max).clamp(-self.action_max, self.action_max), std

    def get_deterministic_action(self, x):
        mu, _ = self.actor(x)
        return mu * self.action_max

    # def get_action(self,x):
    #     return self.actor(x)[0] + torch.tensor(self.noise.sample()).to(self.device), self.actor(x)[1]

    def put_data(self,transition):
        self.data.put_data(transition)
        
    def train_net(self, batch_size, n_epi):
        data = self.data.sample(shuffle = True, batch_size = batch_size)
        states, actions, rewards, next_states, dones = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])

        # targets = rewards + self.args.gamma * (1 - dones) * self.target_q(next_states, self.target_actor(next_states)[0])

        next_action = self.target_actor(next_states)[0].detach()


        noise = (torch.randn_like(next_action) * self.args.policy_noise).clamp(-self.args.noise_clip, self.args.noise_clip)
        noisy_next_action = (next_action + noise).clamp(-1, 1) * self.action_max

        targets = rewards + self.args.gamma * (1 - dones) * self.target_q(next_states, noisy_next_action).detach()

        # q_loss = F.smooth_l1_loss(self.q(states,actions), targets.detach())
        q_loss = F.mse_loss(self.q(states, actions), targets.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=1.0)  # ✅ CLIP HERE
        self.q_optimizer.step()
        
        actor_loss = - self.q(states, self.actor(states)[0]).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # ✅ Optional
        self.actor_optimizer.step()
        
        self.soft_update(self.q, self.target_q, self.args.soft_update_rate)
        self.soft_update(self.actor, self.target_actor, self.args.soft_update_rate)
        if self.writer != None:
            self.writer.add_scalar("loss/q", q_loss, n_epi)
            self.writer.add_scalar("loss/actor", actor_loss, n_epi)