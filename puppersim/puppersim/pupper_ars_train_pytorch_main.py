import os
import sys
import torch
import torch.nn as nn
import gym
from gym import wrappers
import mujoco_py
import pybullet_envs

from pso import PSO
from utils import Normalizer, mkdir

import os
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd

def create_pupper_env():
  CONFIG_DIR = puppersim.getPupperSimPath()+"/"
  _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_pmtg.gin")
#  _NUM_STEPS = 10000
#  _ENV_RANDOM_SEED = 2 
   
  gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
  gin.parse_config_file(_CONFIG_FILE)
  env = env_loader.load()
  return env

# hyper parameters
class Hp():
    def __init__(self):
        #self.main_loop_size = 100
        self.inner_steps = 6
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

# training loop
def inner_train(env,pso, normalizer, hp):
    fitness = []
    for episode in range(hp.inner_steps):
        # init perturbations
        pso.sample()

        # perturbations left
        for k in range(hp.n_directions):
            state = env.reset()
            done = False
            num_plays = 0
            while not done and num_plays<hp.horizon:
                state, reward, done = run(env, pso, normalizer, state, k, 'left')
                num_plays += 1

        # perturbations right
        for k in range(hp.n_directions):
            state = env.reset()
            done = False
            num_plays = 0
            while not done and num_plays<hp.horizon:
                state, reward, done = run(env, pso, normalizer, state, k, 'right')
                num_plays += 1

        # update policy
        pso.update()

        # evaluate
        state = env.reset()
        done = False
        reward_evaluation = 0
        # while not done :
        eval_range = 1000
        for _ in range(eval_range):
            state, reward, done = run(env, pso, normalizer, state)
            reward_evaluation += reward

        # finish, print:
        print('episode',episode,'reward_evaluation',reward_evaluation)
        fitness.append(reward_evaluation)
    return fitness
def outer_train(envs,meta_policy, normalizer, hp, epsilon):
    support_rewards = []
    parameters_shift = {
        k: torch.zeros(v.shape)
        for k, v in meta_policy.state_dict().items()
    }
    
    for env in envs:
        local_policy = deepcopy(meta_policy)
        pso = PSO(local_policy, hp.lr, hp.std, hp.b, hp.n_directions)
        fitness = inner_train(env, pso, normalizer, hp)
        
        parameter_shift = {
            k: v + local_policy.state_dict()[k] - meta_policy.state_dict()[k]
            for k,v in parameter_shift.items()
        }
        support_rewards.append(fitness)
    updated_state = {
        k: v + epsilon * parameter_shift[k]/len(envs)
        for k,v in meta_policy.state_dict().items()
    }
    meta_policy.load_state_dict(updated_state)
    
    return support_rewards
def train(env_dataloader, meta_policy, normalizer, hp, epsilon):
    pre_adapt_average = []
    post_adapt_average = []
    for envs in env_dataloader:
        support_rewards = outer_train(envs, meta_policy, normalizer, hp, epsilon)
        support_rewards = torch.tensor(support_rewards)
        pre_adapt_average.append(torch.mean(support_rewards[:, 0]))
        post_adapt_average.append(torch.mean(support_rewards[:, -1]))
    
        
        
        
    
        
        
if __name__ == '__main__':
    hp = Hp()

    work_dir = mkdir('exp', 'brs')
    # monitor_dir = mkdir(work_dir, 'monitor')
    env = create_pupper_env()
    # gym.make(hp.env_name)

    env.seed(hp.seed)
    torch.manual_seed(hp.seed)
    # env = wrappers.Monitor(env, monitor_dir, force=True)

    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]

    policy = nn.Linear(num_inputs, num_outputs, bias=True)
    policy.weight.data.fill_(0)
    policy.bias.data.fill_(0)

    #pso = PSO(policy, hp.lr, hp.std, hp.b, hp.n_directions)
    normalizer = Normalizer(num_inputs)
    #fitness = train(env, pso, normalizer, hp)

    torch.save(policy.state_dict(), 'model.pt')

    import matplotlib.pyplot as plt
    plt.plot(fitness)
    plt.show()
