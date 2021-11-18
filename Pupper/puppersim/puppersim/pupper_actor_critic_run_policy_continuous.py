"""

Code to load a policy and generate rollout data. Adapted from https://github.com/berkeleydeeprlcourse. 
Example usage:
python3 pupper_actor_critic_run_policy_continuous.py --expert_policy_file=data/lin_policy_plus_best_10.npz --json_file=data/params.json

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

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
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

pi = T.FloatTensor([math.pi])

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.fc3_ = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.fc3(x)
        sigma = self.fc3_(x)

        return mu, sigma


class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=16, 
            layer1_size=64, layer2_size=64, n_outputs=16):
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions)
        self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=1)

    def choose_action(self, observation):
        mu, sigma = self.actor.forward(observation)
        sigma = T.exp(sigma)
        # action_probs = T.distributions.Normal(mu, sigma)
        action_probs = T.distributions.MultivariateNormal(mu, T.diag(sigma))
        # probs = action_probs.sample(sample_shape=T.Size([self.n_outputs]))
        probs = action_probs.sample()
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        # bound actions between values, maybe [-pi / 4, pi / 4]
        # action = pi / 4 * T.tanh(probs)
        action = probs

        return action.detach().numpy()

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        # calculate value of next state by passing in new state through critic network
        critic_value_ = self.critic.forward(new_state)
        # calculate value of current state
        critic_value = self.critic.forward(state)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        # temporal difference loss (1 - done, if episode is over, discards discounted term), i.e. receive no rewards after end of episode
        delta = reward + self.gamma * critic_value_ * (1 - int(done)) - critic_value

        actor_loss = -(1 * self.log_probs * delta).sum()
        critic_loss = delta ** 2

        # print(f'losses actor {actor_loss} critic {critic_loss}')

        (actor_loss + critic_loss).backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
  
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

    agent = Agent(alpha=0.000001, beta=0.000001, input_dims=[14], gamma=0.99,
                layer1_size=256, layer2_size=256)

    agent.actor.load_state_dict(T.load('actor.pt'))
    agent.critic.load_state_dict(T.load('critic.pt'))
  
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
            action = agent.choose_action(obs)
            #action[0:12] = 0
            observations.append(obs)
            actions.append(action)
                        
            # action = action.detach().numpy()
            # print(action[0])
            obs, r, done, _ = env.step(action)
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
