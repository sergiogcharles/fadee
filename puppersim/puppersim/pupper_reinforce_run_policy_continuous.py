"""

Code to load a policy and generate rollout data. Adapted from https://github.com/berkeleydeeprlcourse. 
Example usage:
python3 pupper_reinforce_run_policy.py --expert_policy_file=data/lin_policy_plus_best_10.npz --json_file=data/params.json

"""
import numpy as np
import gym
import time
import pybullet_envs
try:
  import tds_environments
except:
  pass
import json
import time
import os

#temp hack to create an envs_v2 pupper env

import os
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd

# for reinforce

import argparse
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from puppersim.pupper_gym_env import PupperGymEnv
from torch.distributions import MultivariateNormal

from torch.autograd import Variable
import math

def create_pupper_env(args):
  CONFIG_DIR = puppersim.getPupperSimPath()+"/"
  if args.run_on_robot:
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_pmtg_robot.gin")
  else:
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_pmtg.gin")
  gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
  gin.parse_config_file(_CONFIG_FILE)
  gin.bind_parameter("SimulationParameters.enable_rendering", True)
  env = env_loader.load()
  
  return env

pi = Variable(torch.FloatTensor([math.pi]))

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
    def __init__(self, action_space, hidden_size=128, num_inputs=14):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    def select_action(self, state):
        mu, sigma_sq = self.model(Variable(state))
        mu, sigma_sq = self.model(state)
        sigma_sq = F.softplus(sigma_sq)

        eps = torch.randn(mu.size())
        # calculate the probability
        action = (mu + sigma_sq.sqrt()*Variable(eps)).data
        prob = normal(action, mu, sigma_sq)
        entropy = -0.5 * ((sigma_sq + 2 * pi.expand_as(sigma_sq)).log() + 1)

        log_prob = prob.log()
        return action, log_prob, entropy
  
def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', type=str, default="")
    parser.add_argument('--nosleep', action='store_true')

    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert rollouts')
    parser.add_argument('--json_file', type=str, default="")
    parser.add_argument('--run_on_robot', action='store_true')
    parser.add_argument('--render', action='store_true')
    if len(argv):
      args = parser.parse_args(argv)
    else:
      args = parser.parse_args()

    # print('loading and building expert policy')
    # if len(args.json_file)==0:
    #   args.json_file = tp.getDataPath()+"/"+ args.envname+"/params.json"    
    # with open(args.json_file) as f:
    #    params = json.load(f)
    # print("params=",params)
    # if len(args.expert_policy_file)==0:
    #   args.expert_policy_file=tp.getDataPath()+"/"+args.envname+"/nn_policy_plus.npz" 
    #   if not os.path.exists(args.expert_policy_file):
    #     args.expert_policy_file=tp.getDataPath()+"/"+args.envname+"/lin_policy_plus.npz"
    # data = np.load(args.expert_policy_file, allow_pickle=True)

    # print('create gym environment:', params["env_name"])
    env = create_pupper_env(args)#gym.make(params["env_name"])


    # lst = data.files
    # weights = data[lst[0]][0]
    # mu = data[lst[0]][1]
    # print("mu=",mu)
    # std = data[lst[0]][2]
    # print("std=",std)
        
    # ob_dim = env.observation_space.shape[0]
    # ac_dim = env.action_space.shape[0]
    # ac_lb = env.action_space.low
    # ac_ub = env.action_space.high
    
    # policy_params={'type': params["policy_type"],
    #                'ob_filter':params['filter'],
    #                'ob_dim':ob_dim,
    #                'ac_dim':ac_dim,
    #                'action_lower_bound' : ac_lb,
    #                'action_upper_bound' : ac_ub,
    # }
    # policy_params['weights'] = weights
    # policy_params['observation_filter_mean'] = mu
    # policy_params['observation_filter_std'] = std
    # if params["policy_type"]=="nn":
    #   print("FullyConnectedNeuralNetworkPolicy")
    #   policy_sizes_string = params['policy_network_size_list'].split(',')
    #   print("policy_sizes_string=",policy_sizes_string)
    #   policy_sizes_list = [int(item) for item in policy_sizes_string]
    #   print("policy_sizes_list=",policy_sizes_list)
    #   policy_params['policy_network_size'] = policy_sizes_list
    #   policy = FullyConnectedNeuralNetworkPolicy(policy_params, update_filter=False)
    # else:
    #   print("LinearPolicy2")
    #   policy = LinearPolicy2(policy_params, update_filter=False)
    # policy.get_weights()

    agent = REINFORCE(action_space=env.action_space)

    agent.model.load_state_dict(torch.load('model.pt'))
  
    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        # print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        start_time = env.robot.GetTimeSinceReset()
        current_time = start_time
        while not done:
            start_time_robot = current_time
            start_time_wall = time.time()
            # action = policy.act(obs)
            action, _, _ = agent.select_action(torch.tensor([obs]))
            #action[0:12] = 0
            observations.append(obs)
            actions.append(action)
                        
            action = action.detach().numpy()
            # print(action[0])
            obs, r, done, _ = env.step(action[0])
            totalr += r
            steps += 1
            current_time = env.robot.GetTimeSinceReset()
            expected_duration = current_time - start_time_robot
            actual_duration = time.time() - start_time_wall
            if not args.nosleep and actual_duration < expected_duration:
              time.sleep(expected_duration - actual_duration)
            if steps % 10 == 0: 
            	print("Avg time step: ", env.get_time_since_reset() / steps)
            #	print("sim time {}, actual time: {}".format(env.get_time_since_reset(), time.time() - start_time))
            
            #if steps >= env.spec.timestep_limit:
            #    break
        #print("steps=",steps)
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
