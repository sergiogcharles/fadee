import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *
from puppersim.pupper_gym_env import PupperGymEnv
import torch

# env = NormalizedEnv(gym.make("Pendulum-v0"))
# env = PupperGymEnv()
env = gym.make('HalfCheetah-v2') 

agent = DDPGagent(env)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []

for episode in range(1000):
    state = env.reset()
    noise.reset()
    episode_reward = 0
    
    for step in range(100):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action) 
        agent.memory.push(state, action, reward, new_state, done)
        
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        
        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break
    print("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

    torch.save(agent.critic.state_dict(), '../puppersim/puppersim/critic.pt')
    torch.save(agent.actor.state_dict(), '../puppersim/puppersim/actor.pt')


plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()