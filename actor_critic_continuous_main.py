import numpy as np
import torch as T
import gym
from actor_critic_continuous import Agent 
from puppersim.pupper_gym_env import PupperGymEnv
import matplotlib.pyplot as plt

import os
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd

def create_pupper_env():
  CONFIG_DIR = puppersim.getPupperSimPath()+"/"
  _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper.gin")
#  _NUM_STEPS = 10000
#  _ENV_RANDOM_SEED = 2 
   
  gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
  gin.parse_config_file(_CONFIG_FILE)
  env = env_loader.load()
  return env

if __name__ == '__main__':
    # env = gym.make('HalfCheetah-v2') 
    env = create_pupper_env()
    
    agent = Agent(alpha=0.000001, beta=0.000001, input_dims=[env.observation_space.shape[0]], gamma=0.99,
                layer1_size=256, layer2_size=256, n_outputs=env.action_space.shape[0])

    # env = PupperGymEnv()
    # score_history = []

    rewards = []
    avg_rewards = []

    num_episodes = 1000
    for episode in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            loss = agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward
        # score_history.append(score)
        print("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(score, decimals=2), np.mean(rewards[-10:])))
        rewards.append(score)
        avg_rewards.append(np.mean(rewards[-10:]))

        T.save(agent.actor.state_dict(), 'puppersim/puppersim/actor.pt')
        T.save(agent.critic.state_dict(), 'puppersim/puppersim/critic.pt')
    

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
    
