import sys
import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable

import argparse, math, os
import numpy as np
import gym
from gym import wrappers

from puppersim.pupper_gym_env import PupperGymEnv

import os
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd


pi = Variable(torch.FloatTensor([math.pi]))

# def create_pupper_env():
#   CONFIG_DIR = puppersim.getPupperSimPath()+"/"
#   _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper.gin")
# #  _NUM_STEPS = 10000
# #  _ENV_RANDOM_SEED = 2 
   
#   gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
#   gin.parse_config_file(_CONFIG_FILE)
#   env = env_loader.load()
#   return env

def normal(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b


class Policy(nn.Module):
    def __init__(self, hidden_size=128, num_inputs=14, action_space=16):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.hidden_linear1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_linear2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_linear3 = nn.Linear(hidden_size, hidden_size)

        self.linear2 = nn.Linear(hidden_size, num_outputs)
        self.linear2_ = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        x = F.relu(self.hidden_linear1(x))
        x = F.relu(self.hidden_linear2(x))
        x = F.relu(self.hidden_linear3(x))
        mu = self.linear2(x)
        sigma_sq = self.linear2_(x)

        return mu, sigma_sq


class REINFORCE:
    def __init__(self, hidden_size=128, num_inputs=14, action_space=16):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
        self.model.train()

    def select_action(self, state):
        mu, sigma_sq = self.model(Variable(state))
        # mu = mu.clamp(-pi/8, pi/8)
        # sigma_sq = F.softplus(sigma_sq + 1e-5)
        sigma_sq = torch.exp(sigma_sq)
        # sigma_sq = sigma_sq.clamp(1e-5, 0.1)

        eps = torch.randn(mu.size())
        # calculate the probability
        action = (mu + sigma_sq.sqrt()*Variable(eps)).data
        # action = pi / 4 * torch.tanh(action)
        prob = normal(action, mu, sigma_sq)
        entropy = -0.5 * ((sigma_sq + 2 * pi.expand_as(sigma_sq)).log() + 1)

        log_prob = prob.log()
        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i]))).sum() - (0.0001*entropies[i]).sum()
        loss = loss / len(rewards)
		
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env_name', type=str, default='CartPole-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=10000, metavar='N',
                    help='number of episodes (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--ckpt_freq', type=int, default=100, 
		    help='model saving frequency')
parser.add_argument('--display', type=bool, default=False,
                    help='display or not')
args = parser.parse_args()

# env_name = args.env_name
# env = gym.make(env_name)
env = PupperGymEnv()
# env = create_pupper_env()

# if type(env.action_space) != gym.spaces.discrete.Discrete:
#     from reinforce_continuous import REINFORCE
#     env = NormalizedActions(gym.make(env_name))
# else:
#     from reinforce_discrete import REINFORCE

# if args.display:
#     env = wrappers.Monitor(env, '/tmp/{}-experiment'.format(env_name), force=True)

# env.seed(args.seed)
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)

agent = REINFORCE(args.hidden_size, env.observation_space.shape[0], env.action_space)

for i_episode in range(args.num_episodes):
    state = torch.Tensor([env.reset()])
    entropies = []
    log_probs = []
    rewards = []
    for t in range(args.num_steps):
        action, log_prob, entropy = agent.select_action(state)
        action = action.cpu()

        next_state, reward, done, _ = env.step(action.numpy()[0])

        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = torch.Tensor([next_state])

        if done:
            # print(f'Episode {i_episode} Time-step {t} Reward {reward}')
            break

    agent.update_parameters(rewards, log_probs, entropies, args.gamma)

    print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))

    if i_episode%args.ckpt_freq == 0:
        torch.save(agent.model.state_dict(), 'puppersim/puppersim/model.pt')
        # print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
	
env.close()