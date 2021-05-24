import numpy as np
import torch
import gym
import pybulletgym
import argparse
import os
import utils
import random
import math
import time

import TD3 ## baselines
import DDPG ## baselines
import DARC
import DADDPG
import DATD3
from tensorboardX import SummaryWriter


def eval_policy(policy, env_name, seed, eval_episodes=10, eval_cnt=None):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for episode_idx in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			next_state, reward, done, _ = eval_env.step(action)

			avg_reward += reward
			state = next_state
	avg_reward /= eval_episodes

	print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))
	
	return avg_reward


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dir", default="./logs")
	parser.add_argument("--policy", default="MAC", help='policy to use, support MAC, TD3')
	parser.add_argument("--env", default="HalfCheetah-v2")
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument("--start-steps", default=1e4, type=int, help='Number of steps for the warm-up stage using random policy')
	parser.add_argument("--eval-freq", default=5000, type=int, help='Number of steps per evaluation')
	parser.add_argument("--steps", default=1e6, type=int, help='Maximum number of steps')

	parser.add_argument("--discount", default=0.99, help='Discount factor')
	parser.add_argument("--tau", default=0.005, help='Target network update rate')

	parser.add_argument("--qweight", default=0.2, type=float, help='The weighting coefficient that correlates value estimation from double actors')
	parser.add_argument("--reg", default=0.005, type=float, help='The regularization parameter for DARC')               
	
	parser.add_argument("--actor-lr", default=1e-3, type=float)     
	parser.add_argument("--critic-lr", default=1e-3, type=float)    
	parser.add_argument("--hidden-sizes", default='400,300', type=str)  
	parser.add_argument("--batch-size", default=100, type=int)      # Batch size for both actor and critic

	parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load-model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

	parser.add_argument("--expl-noise", default=0.1, type=float)                # Std of Gaussian exploration noise
	parser.add_argument("--policy-noise", default=0.2, type=float)              # Noise added to target policy during critic update
	parser.add_argument("--noise-clip", default=0.5, type=float)                # Range to clip target policy noise

	parser.add_argument("--policy-freq", default=2, type=int, help='Frequency of delayed policy updates')
	
	args = parser.parse_args()

	print("------------------------------------------------------------")
	print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
	print("------------------------------------------------------------")
	
	outdir = args.dir
	writer = SummaryWriter('{}/tb'.format(outdir))
	if args.save_model and not os.path.exists("{}/models".format(outdir)):
		os.makedirs("{}/models".format(outdir))

	env = gym.make(args.env)

	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.cuda.manual_seed_all(args.seed)
	random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	min_action = -max_action
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"hidden_sizes": [int(hs) for hs in args.hidden_sizes.split(',')],
		"actor_lr": args.actor_lr,
		"critic_lr": args.critic_lr,
		"device": device,
	}

	if args.policy == "TD3":
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq

		policy = TD3.TD3(**kwargs)

	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)
	
	elif args.policy == "DATD3":
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action

		policy = DATD3.DATD3(**kwargs)
	
	elif args.policy == "DARC":
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["q_weight"] = args.qweight
		kwargs["regularization_weight"] = args.reg

		policy = DARC.DARC(**kwargs)
	
	elif args.policy == "DADDPG":
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action

		policy = DADDPG.DADDPG(**kwargs)

	if args.load_model != "":
		policy.load("./models/{}".format(args.load_model))
	
	## write logs to record training parameters
	with open(outdir + 'log.txt','w') as f:
		f.write('\n Policy: {}; Env: {}, seed: {}'.format(args.policy, args.env, args.seed))
		for item in kwargs.items():
			f.write('\n {}'.format(item))

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

	eval_cnt = 0
	
	eval_return = eval_policy(policy, args.env, args.seed, eval_cnt=eval_cnt)
	eval_cnt += 1

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.steps)):
		episode_timesteps += 1

		# select action randomly or according to policy
		if t < args.start_steps:
			action = (max_action - min_action) * np.random.random(env.action_space.shape) + min_action
		else:
			action = (
				policy.select_action(np.array(state)) + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward
		if t >= args.start_steps:
			policy.train(replay_buffer, args.batch_size)
		
		if done: 
			print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, episode_num+1, episode_timesteps, episode_reward))
			writer.add_scalar('train return', episode_reward, global_step = t+1)

			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		if (t + 1) % args.eval_freq == 0:
			eval_return = eval_policy(policy, args.env, args.seed, eval_cnt=eval_cnt)
			writer.add_scalar('test return', eval_return, global_step = t+1)
			eval_cnt += 1

			if args.save_model:
				policy.save('{}/models/model'.format(outdir))
	writer.close()
