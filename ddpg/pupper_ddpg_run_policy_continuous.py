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

import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from models import *
from utils import *
from ddpg import DDPGagent


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

    env = create_pupper_env(args)
    agent = DDPGagent(env)
    
    agent.actor.load_state_dict(torch.load('../puppersim/puppersim/actor.pt'))
    agent.critic.load_state_dict(torch.load('../puppersim/puppersim/critic.pt'))
  
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
            action = agent.get_action(obs)
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
