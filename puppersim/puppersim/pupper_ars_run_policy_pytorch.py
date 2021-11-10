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

from pso import PSO
from utils import Normalizer, mkdir

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

# hyper parameters
class Hp():
    def __init__(self):
        self.main_loop_size = 100
        self.horizon = 1000
        self.lr = 0.02
        self.n_directions = 8
        self.b = 8
        assert self.b<=self.n_directions, "b must be <= n_directions"
        self.std = 0.03
        self.seed = 1
        ''' chose your favourite '''
        #self.env_name = 'Reacher-v1'
        #self.env_name = 'Pendulum-v0'
        #self.env_name = 'HalfCheetahBulletEnv-v0'
        #self.env_name = 'Hopper-v1'#'HopperBulletEnv-v0'
        #self.env_name = 'Ant-v1'#'AntBulletEnv-v0'#
        self.env_name = 'HalfCheetah-v1'
        #self.env_name = 'Swimmer-v1'
        #self.env_name = 'Humanoid-v1'

def run(env, pso, normalizer, state, direction=None, side='left'):
    normalizer.observe(state)
    state = normalizer.normalize(state)
    state = torch.from_numpy(state).float()
    action = pso.evaluate(state, direction, side).numpy()
    state, reward, done, _ = env.step(action)
    reward = max(min(reward, 1), -1)
    if direction is not None:
        pso.reward(reward, direction, side)
    return state, reward, done
  
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

    env = create_pupper_env(args)#gym.make(params["env_name"])

    hp = Hp()
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    print(env.observation_space.shape[0])

    policy = nn.Linear(num_inputs, num_outputs, bias=True)
    policy.load_state_dict(torch.load('model.pt'))

    pso = PSO(policy, hp.lr, hp.std, hp.b, hp.n_directions)
    normalizer = Normalizer(num_inputs)
  
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
            action, _, _ = run(env, pso, normalizer, obs)
            # agent.select_action(obs)
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
