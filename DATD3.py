import copy
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_sizes=[400, 300]):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_sizes=[400, 300]):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], 1)


	def forward(self, state, action):
		if len(state.shape) == 3:
			sa = torch.cat([state, action], 2)
		else:
			sa = torch.cat([state, action], 1)

		q = F.relu(self.l1(sa))
		q = F.relu(self.l2(q))
		q = self.l3(q)

		return q


class DATD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		device,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		actor_lr=1e-3,
		critic_lr=1e-3,
		hidden_sizes=[400, 300],
	):
		self.device = device

		self.actor1 = Actor(state_dim, action_dim, max_action, hidden_sizes).to(self.device)
		self.actor1_target = copy.deepcopy(self.actor1)
		self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=actor_lr)

		self.actor2 = Actor(state_dim, action_dim, max_action, hidden_sizes).to(self.device)
		self.actor2_target = copy.deepcopy(self.actor2)
		self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=actor_lr)

		self.critic1 = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
		self.critic1_target = copy.deepcopy(self.critic1)
		self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)

		self.critic2 = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
		self.critic2_target = copy.deepcopy(self.critic2)
		self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

		action1 = self.actor1(state)
		action2 = self.actor2(state)

		q1 = self.critic1(state, action1)
		q2 = self.critic2(state, action2)

		action = action1 if q1 >= q2 else action2

		return action.cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100):
		## cross update scheme
		self.train_one_q_and_pi(replay_buffer, True, batch_size=batch_size)
		self.train_one_q_and_pi(replay_buffer, False, batch_size=batch_size)

	def train_one_q_and_pi(self, replay_buffer, update_a1 = True, batch_size=100):

		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			next_action1 = self.actor1_target(next_state)
			next_action2 = self.actor2_target(next_state)

			noise = torch.randn(
				(action.shape[0], action.shape[1]), 
				dtype=action.dtype, layout=action.layout, device=action.device
			) * self.policy_noise
			noise = noise.clamp(-self.noise_clip, self.noise_clip)

			next_action1 = (next_action1 + noise).clamp(-self.max_action, self.max_action)
			next_action2 = (next_action2 + noise).clamp(-self.max_action, self.max_action)

			next_Q1_a1 = self.critic1_target(next_state, next_action1)
			next_Q2_a1 = self.critic2_target(next_state, next_action1)

			next_Q1_a2 = self.critic1_target(next_state, next_action2)
			next_Q2_a2 = self.critic2_target(next_state, next_action2)
			## min first, max afterward to avoid underestimation bias
			next_Q1 = torch.min(next_Q1_a1, next_Q2_a1)
			next_Q2 = torch.min(next_Q1_a2, next_Q2_a2)

			next_Q = torch.max(next_Q1, next_Q2)
			target_Q = reward + not_done * self.discount * next_Q

		if update_a1:
			current_Q1 = self.critic1(state, action)
			critic1_loss = F.mse_loss(current_Q1, target_Q)

			self.critic1_optimizer.zero_grad()
			critic1_loss.backward()
			self.critic1_optimizer.step()

			actor1_loss = -self.critic1(state, self.actor1(state)).mean()
			
			self.actor1_optimizer.zero_grad()
			actor1_loss.backward()
			self.actor1_optimizer.step()

			for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor1.parameters(), self.actor1_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		else:
			current_Q2 = self.critic2(state, action)
			critic2_loss = F.mse_loss(current_Q2, target_Q)

			self.critic2_optimizer.zero_grad()
			critic2_loss.backward()
			self.critic2_optimizer.step()

			actor2_loss = -self.critic2(state, self.actor2(state)).mean()
			
			self.actor2_optimizer.zero_grad()
			actor2_loss.backward()
			self.actor2_optimizer.step()
			
			for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor2.parameters(), self.actor2_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, filename):
		torch.save(self.critic1.state_dict(), filename + "_critic1")
		torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")
		torch.save(self.actor1.state_dict(), filename + "_actor1")
		torch.save(self.actor1_optimizer.state_dict(), filename + "_actor1_optimizer")

		torch.save(self.critic2.state_dict(), filename + "_critic2")
		torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")
		torch.save(self.actor2.state_dict(), filename + "_actor2")
		torch.save(self.actor2_optimizer.state_dict(), filename + "_actor2_optimizer")

	def load(self, filename):
		self.critic1.load_state_dict(torch.load(filename + "_critic1"))
		self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer"))
		self.actor1.load_state_dict(torch.load(filename + "_actor1"))
		self.actor1_optimizer.load_state_dict(torch.load(filename + "_actor1_optimizer"))

		self.critic2.load_state_dict(torch.load(filename + "_critic2"))
		self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer"))
		self.actor2.load_state_dict(torch.load(filename + "_actor2"))
		self.actor2_optimizer.load_state_dict(torch.load(filename + "_actor2_optimizer"))