import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

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
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=6, 
            layer1_size=64, layer2_size=64, n_outputs=6):
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

        actor_loss = -(1 * self.log_probs.mean() * delta)
        critic_loss = delta ** 2

        # print(f'losses actor {actor_loss} critic {critic_loss}')

        (actor_loss + critic_loss).backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

        return actor_loss + critic_loss

