import numpy as np
import torch as T
import gym
from actor_critic_continuous import Agent 
from puppersim.pupper_gym_env import PupperGymEnv


if __name__ == '__main__':
    agent = Agent(alpha=0.000005, beta=0.000001, input_dims=[14], gamma=0.99,
                layer1_size=256, layer2_size=256)

    env = PupperGymEnv()

    score_history = []
    num_episodes = 1000
    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward
        score_history.append(score)
        print(f'episode {i} score {score}')
        T.save(agent.actor.state_dict(), 'puppersim/puppersim/actor.pt')
        T.save(agent.critic.state_dict(), 'puppersim/puppersim/critic.pt')
