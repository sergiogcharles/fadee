
import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from puppersim.pupper_gym_env import PupperGymEnv
from torch.distributions import MultivariateNormal
num_samples_per_training_step = 50

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


# env =  gym.make('CartPole-v1')
env = PupperGymEnv()
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(14, 128)
        self.dropout = nn.Dropout(p=0.1)
        self.affine2 = nn.Linear(128, 16)
        self.affine2_ = nn.Linear(128, 16)

        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        # action_scores = self.affine2(x)
        #action_scores = F.tanh(action_scores)

        mu = self.affine2(x)
        sigma_sq = self.affine2_(x)

        return mu, sigma_sq

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    mu, sigma_sq = policy(state)
    mu = mu.reshape(-1)
    sigma_sq = sigma_sq.reshape(-1)
    # mu, sigma = torch.tanh(normal_params[0]), torch.diag(torch.sigmoid(normal_params[1]))
    sigma_sq = F.softplus(sigma_sq)
    dist = MultivariateNormal(mu, torch.diag(sigma_sq))
    action = dist.sample()
    #policy.saved_log_probs.append(dist.log_prob(action))
    return action.detach().numpy(), dist.log_prob(action)

def finish_episode():
    policy_loss = []
    for trajectory_rewards, trajectory_log_probs in zip(policy.rewards, policy.log_probs):
        #discount our rewards
        R = 0
        returns_trajectory = []
        for r in trajectory_rewards[::-1]:
            R = r + args.gamma * R
            returns_trajectory.insert(0, R)
        returns_trajectory = torch.tensor(returns_trajectory)
        returns_trajectory = (returns_trajectory - returns_trajectory.mean()) / (returns_trajectory.std() + eps)
        #weight our actions for our policy gradient
        for log_prob, R in zip(trajectory_log_probs, returns_trajectory):
            policy_loss.append(-log_prob.reshape(-1) * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum() / num_samples_per_training_step
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.log_probs[:]

    # R = 0
    # policy_loss = []
    # returns = []
    # for r in policy.rewards[::-1]:
    #     R = r + args.gamma * R
    #     returns.insert(0, R)
    # returns = torch.tensor(returns)
    # returns = (returns - returns.mean()) / (returns.std() + eps)
    # for log_prob, R in zip(policy.saved_log_probs, returns):
    #     # print(f'R {R}, log_prob {log_prob}')
    #     policy_loss.append(-log_prob.reshape(-1) * R)
    # optimizer.zero_grad()
    # policy_loss = torch.cat(policy_loss).sum()
    # policy_loss.backward()
    # optimizer.step()
    # del policy.rewards[:]
    # del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):

        for i in range(num_samples_per_training_step):
            trajectory_reward = []
            trajectory_log_probs = []
            state, ep_reward = env.reset(), 0
            for t in range(1, 10000):  # Don't infinite loop while learning
                action, log_prob = select_action(state)
                state, reward, done, _ = env.step(action)
                if args.render:
                    env.render()
                #policy.rewards.append(reward)
                trajectory_reward.append(reward)
                trajectory_log_probs.append(log_prob)
                ep_reward += reward
                if done:
                    break
            #trajectory has been rolled out
            policy.rewards.append(trajectory_reward) #appending a list of rewards
            policy.log_probs.append(trajectory_log_probs) #appending a list of log_probs
            #print(trajectory_reward)
            print(f"trajectory {i} finished")


        ep_reward = ep_reward / num_samples_per_training_step
        running_reward = 0.05 * (ep_reward) + (1 - 0.05) * running_reward
        print("step finished")
        finish_episode()

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.10f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
            # save policy after each log interval
            torch.save(policy.state_dict(), 'puppersim/puppersim/model.pt')
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


if __name__ == '__main__':
    main()