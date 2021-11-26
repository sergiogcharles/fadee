import collections
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import utils as torch_utils
import torch.nn.functional as F
import schedule
import replay
import embed
import utils

# Want: task in n transitions (context) from n timesteps
# Output belief of what z is
# We also mix in off-policy data from sampler
class InferenceNetworkRNN(nn.Module):
    def __init__(self, env, hidden_dim=128, latent_dim=8):
        super(InferenceNetworkRNN, self).__init__()
        # input size = size of 1 context (s, a, r, s') tuple
        context_dim = 2 * env.observation_space.shape[0] + env.action_space.shape[0] + 1

        self.rnn = nn.RNN(input_size=context_dim, hidden_size=hidden_dim, num_layers=3)
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, context):
        # contexts shape = (L, N, H_in) = (n, batch_size, size(concatenated(s, a, r, s'))) for n context in contexts
        # H_in = 2 * state.shape[0] + action.shape[0] + 1
        # output = (n, batch_size, H_out)

        output, hn = self.rnn(context)
        output = F.relu(output)
        z = self.proj(output)

        return z

# class ContextEncoder(nn.Module):
#   def __init__(self, input_dim, output_dim, hidden_dim=128):
#     super(ContextEncoder, self).__init__()

#     self.linear1 = nn.Linear(input_dim, hidden_dim)
#     self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#     self.linear3 = nn.Linear(hidden_dim, output_dim)

#   def forward(self, x):
#     x = self.linear1(x)
#     x = F.relu(x)
#     x = self.linear2(x)
#     x = F.relu(x)
#     x = self.linear3(x)

#     return x

class InferenceNetwork(nn.Module):
  def __init__(self, env, exploration, hidden_dim=128, latent_dim=8):
    super(InferenceNetwork, self).__init__()

    self.exploration = exploration

    observation_space_size = 0
    if self.exploration:
      # print(env.observation_space.spaces["observation"].shape)
      observation_space_size = env.observation_space.spaces["observation"].shape[0]
    else:
      for key, value in env.observation_space.spaces.items():
        if key == "observation" or key == "instructions":
          observation_space_size += value.shape[0]

    # action_space_size = env.action_space.n
    # print(observation_space_size)
    transition_input_dim = 2 * observation_space_size + 1 + 1
    # print(f'obs size {2 * observation_space_size + action_space_size + 1}')

    self.linear1 = nn.Linear(transition_input_dim, hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = nn.Linear(hidden_dim, hidden_dim)

    # Get f_theta^mu, f_theta^sigma
    self.mu_head = nn.Linear(hidden_dim, latent_dim)
    self.sigma_head = nn.Linear(hidden_dim, latent_dim)

  def forward(self, context):
    # breakpoint()
    x = self.linear1(context)
    x = F.relu(x)
    x = self.linear2(x)
    x = F.relu(x)
    x = self.linear3(x)
    x = F.relu(x)

    mu = self.mu_head(x)
    sigma_squared = F.softplus(self.sigma_head(x))

    return mu, sigma_squared

  # def sample_latent(self, contexts):
  #   mu, sigma = self.forward(context)
  #   Psi_posterior = torch.distributions.Normal(mu, sigma)
  #   # Psi_posterior = T.distributions.MultivariateNormal(mu, T.diag(sigma))
  #   # probs = action_probs.sample(sample_shape=T.Size([self.n_outputs]))
  #   z = Psi_posterior.sample()

  #   return z


class DQNAgent(object):
  @classmethod
  def from_config(cls, config, env, exploration):
    dqn = DQNPolicy.from_config(config.get("policy"), env, exploration)
    replay_buffer = replay.ReplayBuffer.from_config(config.get("buffer"))
    optimizer_dqn = optim.Adam(dqn.parameters(), lr=config.get("learning_rate"))
    inference_net = InferenceNetwork(env, exploration)
    optimizer_inference = optim.Adam(inference_net.parameters(), lr=config.get("learning_rate"))
    latent_dim = 8

    return cls(dqn, replay_buffer, optimizer_dqn, optimizer_inference, config.get("sync_target_freq"),
               config.get("min_buffer_size"), config.get("batch_size"),
               config.get("update_freq"), config.get("max_grad_norm"), inference_net, exploration, latent_dim)

  def __init__(self, dqn, replay_buffer, optimizer_dqn, optimizer_inference, sync_freq,
               min_buffer_size, batch_size, update_freq, max_grad_norm, inference_net, exploration, latent_dim=8):
    """
    Args:
      dqn (DQNPolicy)
      replay_buffer (ReplayBuffer)
      optimizer_dqn (torch.Optimizer)
      optimizer_inference (torch.Optimizer)
      sync_freq (int): number of updates between syncing the
        DQN target Q network
      min_buffer_size (int): replay buffer must be at least this large
        before taking grad updates
      batch_size (int): number of experience to sample per grad step
      update_freq (int): number of update calls per parameter update.
      max_grad_norm (float): gradient is clipped to this norm on each
        update
    """
    self._dqn = dqn
    self._replay_buffer = replay_buffer
    self._optimizer_dqn = optimizer_dqn
    self._optimizer_inference = optimizer_inference
    self._sync_freq = sync_freq
    self._min_buffer_size = min_buffer_size
    self._batch_size = batch_size
    self._update_freq = update_freq
    self._max_grad_norm = max_grad_norm
    self._updates = 0
    self._exploration = exploration


    # Torch DQN losses
    self._dqn_losses = collections.deque(maxlen=100)
    # Integer losses for everything else
    self._dqn_theta_losses = collections.deque(maxlen=100)
    self._kl_phi_losses = collections.deque(maxlen=100)
    self._total_phi_losses = collections.deque(maxlen=100)
    self._losses = collections.deque(maxlen=100)
    self._grad_norms = collections.deque(maxlen=100)

    self._inference_net = inference_net
    self.z = None
    self.context_mu = None
    self.context_sigma_squared = None
    self._latent_dim = latent_dim
    self.context_batch_size = 10

  def product_of_guassians(self, mus, sigmas_squared):
    # Compute mu, sigma_squared for product of Gaussians 
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)

    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)

    return mu, sigma_squared

  def infer_posterior(self, context):
    if context != None:
      # Compute q(z|c) from context then sample z from q(z|c)
      mus, sigmas_squared = self._inference_net(context)
      # #transitions in context x latent_dim
      # should return latent_dim-dimensional mu, sigma_squared
      self.context_mu, self.context_sigma_squared = self.product_of_guassians(mus, sigmas_squared)

      dist = torch.distributions.Normal(self.context_mu, self.context_sigma_squared)

      self.z = dist.sample().reshape(1, -1)
    else:
      # Default to prior if no context
      dist = torch.distributions.Normal(torch.zeros(self._latent_dim), torch.ones(self._latent_dim))
      self.z = dist.sample().reshape(1, -1)
    return self.z

  def detach_z(self):
    # removes z from computational graph so we don't backprop through
    self.z = self.z.detach()

  def update(self, experience, beta=1e-4):
    """Updates agent on this experience.

    Args:
      experience (Experience): experience to update on.
    """
    # self._replay_buffer.add(experience)

    # If timestep % update_inference_freq
    # if timestep % self._update_inference_freq == 0:
    #   # Sample n recent transitions (i.e. context) from replay buffer

    #   # Sample a min of 5 past steps of episode (if maxed out, all the past steps)
    #   context = self._replay_buffer.context_sample(self._update_inference_freq)

    #   self._optimizer_inference.zero_grad()

    #   # Compute KL-divergence loss
    #   prior = torch.distribution.Normal(torch.zeros(self._latent_dim), torch.ones(self._latent_dim))
    #   # posteriors = [torch.distributions.Normal(mu, sigma) for mu, sigma in zip(torch.unbind(self.context_mu), torch.unbind(self.context_sigma_squared))]
    #   posterior = torch.distributions.Normal(self.context_mu, self.context_sigma_squared)

    #   # kl_losses = [torch.distributions.kl.kl_divergence(prior, posterior) for posterior in posteriors]
    #   # kl_loss = torch.sum(torch.stack(kl_losses))
    #   kl_loss = torch.distributions.kl.kl_divergence(prior, posterior)

    #   kl_loss.backward()
    #   self._kl_losses.append(kl_loss.item())

    #   self._optimizer_inference.step()

    # TODO: NEED TO ENSURE THIS MAKES SENSE

    if len(self._replay_buffer) >= self._min_buffer_size:
      if self._updates % self._update_freq == 0:
        # Sample context and batch (Page 5 of PEARL)

        # This is like batch b^i in 
        experiences = self._replay_buffer.sample(self._batch_size)

        # Sample a min of 5 past steps of episode (if maxed out, all the past steps)
        # context = None
        # if len(self._replay_buffer._storage) < self.context_batch_size:
        #   context = self._replay_buffer.sample_context(len(self._replay_buffer._storage))
        # else:
        #   context = self._replay_buffer.sample_context(self.context_batch_size)
        context = self._replay_buffer.sample_context(len(self._replay_buffer._storage))[0]

        # Update wrt phis
        self._optimizer_inference.zero_grad()

        batch_size = len(context)
        states = [e.state for e in context]
        actions = torch.tensor([e.action for e in context]).long()
        next_states = [e.next_state for e in context]
        rewards = torch.tensor([e.reward for e in context]).float()

        context_list = []

        for b in range(batch_size):
          # index 0 corresponds to "observation", index 1 corresponds to "instructions"
          # instructions returns numpy array, so we need to cast to Long tensor
          # print(torch.cat((next_states[b][0], torch.from_numpy(next_states[b][1]))))
          state = None
          next_state = None
          if self._exploration:
            # Only use observation
            state = states[b][0]
            next_state = next_states[b][0]
          else:
            # Use observation and instructions
            state = torch.cat((states[b][0], torch.from_numpy(states[b][1])))
            next_state = torch.cat((next_states[b][0], torch.from_numpy(next_states[b][1])))

          # Actions
          action = actions[b].unsqueeze(dim=0).type(torch.LongTensor)
          reward = rewards[b].unsqueeze(dim=0).type(torch.LongTensor)

          # Add experience to context
          # print(state, action, reward, next_state)
          # print(state.shape)
          # print(next_state.shape)
          context_list.append(torch.cat((state, action, reward, next_state)))

        # Convert to torch tensor of size batch_size x concatenated experience vector
        context = torch.stack(context_list).type(torch.FloatTensor)

        # Sample z from posterior
        z = self.infer_posterior(context)

        posterior = torch.distributions.Normal(self.context_mu, self.context_sigma_squared)

        # Compute DQN loss
        # breakpoint()
        dqn_loss = self._dqn.loss(experiences, np.ones(self._batch_size), z.detach())        
        self._dqn_losses.append(dqn_loss)
        self._losses.append(dqn_loss.item())

        # Compute KL-divergence loss
        prior = torch.distributions.Normal(torch.zeros(self._latent_dim), torch.ones(self._latent_dim))
        # posteriors = [torch.distributions.Normal(mu, sigma) for mu, sigma in zip(torch.unbind(self.context_mu), torch.unbind(self.context_sigma_squared))]

        # kl_losses = [torch.distributions.kl.kl_divergence(prior, posterior) for posterior in posteriors]
        # kl_loss = torch.sum(torch.stack(kl_losses))
        kl_losses = torch.distributions.kl.kl_divergence(prior, posterior)
        kl_loss = torch.sum(kl_losses)
        # print(f'kl loss {beta * kl_loss} dqn loss {dqn_loss}')
        
        self._kl_phi_losses.append(kl_loss.item())

        total_loss = beta * kl_loss + dqn_loss
        self._total_phi_losses.append(total_loss.item())

        total_loss.backward(retain_graph=True)
        self._optimizer_inference.step()

        ##########################################################################

        # # This is like batch b^i in 
        # experiences = self._replay_buffer.sample(self._batch_size)
        # Update wrt thetas of DQN

        self._optimizer_dqn.zero_grad()
        # Critic loss (if we were in actor-critic setup)
        # loss = self._dqn.loss(experiences, np.ones(self._batch_size), z.detach())
        loss = self._dqn_losses[-1]
        loss.backward()
        self._dqn_theta_losses.append(loss.item())

        # clip according to the max allowed grad norm
        grad_norm = torch_utils.clip_grad_norm_(
            self._dqn.parameters(), self._max_grad_norm, norm_type=2)
        self._grad_norms.append(grad_norm)
        self._optimizer_dqn.step()

        if self._updates % 1000 == 0:
          print(f'Update phi loss: {self._total_phi_losses[-1]}, theta loss: {self._dqn_theta_losses[-1]}')

      if self._updates % self._sync_freq == 0:
        self._dqn.sync_target()

    self._updates += 1

  def act(self, state, z, prev_hidden_state=None, test=False):
    """Given the current state, returns an action.

    Args:
      state (State)

    Returns:
      action (int)
      hidden_state (object)
    """
    return self._dqn.act(state, z, prev_hidden_state=prev_hidden_state, test=test)

  @property
  def stats(self):
    def mean_with_default(l, default):
      if len(l) == 0:
        return default
      return np.mean(l)

    stats = self._dqn.stats
    stats["loss"] = mean_with_default(self._losses, None)
    stats["grad_norm"] = mean_with_default(self._grad_norms, None)
    return {"DQN/{}".format(k): v for k, v in stats.items()}

  def state_dict(self):
    """Returns a serializable dictionary containing all the relevant
    details from the class.

    Returns:
      state_dict (dict)
    """
    # Currently doesn't serialize replay buffer to save memory
    return {
        "dqn": self._dqn.state_dict(),
        #"replay_buffer": self._replay_buffer,
        "optimizer_dqn": self._optimizer_dqn.state_dict(),
        "optimizer_inference": self._optimizer_inference.state_dict(),
        "sync_freq": self._sync_freq,
        "min_buffer_size": self._min_buffer_size,
        "batch_size": self._batch_size,
        "update_freq": self._update_freq,
        "max_grad_norm": self._max_grad_norm,
        "updates": self._updates,
    }

  def load_state_dict(self, state_dict):
    self._dqn.load_state_dict(state_dict["dqn"])
    #self._replay_buffer = state_dict["replay_buffer"]
    self._optimizer_dqn.load_state_dict(state_dict["optimizer_dqn"])
    self._optimizer_inference.load_state_dict(state_dict["optimizer_inference"])
    self._sync_freq = state_dict["sync_freq"]
    self._min_buffer_size = state_dict["min_buffer_size"]
    self._batch_size = state_dict["batch_size"]
    self._update_freq = state_dict["update_freq"]
    self._max_grad_norm = state_dict["max_grad_norm"]
    self._updates = state_dict["updates"]

  def set_reward_relabeler(self, reward_relabeler):
    """See DQNPolicy.reward_relabeler."""
    self._dqn.set_reward_relabeler(reward_relabeler)


# TODO(evzliu): Add Policy base class
class DQNPolicy(nn.Module):
  @classmethod
  def from_config(cls, config, env, exploration):
    def embedder_factory():
      embedder_config = config.get("embedder")
      embed_type = embedder_config.get("type")
      if embed_type == "instruction":
        return embed.InstructionPolicyEmbedder.from_config(embedder_config, env)
      elif embed_type == "recurrent":
        return embed.RecurrentStateEmbedder.from_config(embedder_config, env)
      elif embedder_config.get("type") == "varibad":
        return embed.VariBADEmbedder.from_config(embedder_config, env)
      elif embedder_config.get("type") == "import":
        return embed.RecurrentAndTaskIDEmbedder.from_config(
            embedder_config, env)
      else:
        raise ValueError("Unsupported embedding type: {}".format(embed_type))

    policy_type = config.get("type")
    if policy_type == "vanilla":
      pass
    elif policy_type == "recurrent":
      cls = RecurrentDQNPolicy
    else:
      raise ValueError("Unsupported policy type: {}".format(policy_type))

    epsilon_schedule = schedule.LinearSchedule.from_config(
        config.get("epsilon_schedule"))
    return cls(env.action_space.n, epsilon_schedule, config.get("test_epsilon"),
               embedder_factory, exploration, config.get("discount"))

  def __init__(self, num_actions, epsilon_schedule, test_epsilon,
               state_embedder_factory, exploration, gamma=0.99):
    """DQNPolicy should typically be constructed via from_config, and not
    through the constructor.

    Args:
      num_actions (int): the number of possible actions to take at each
        state
      epsilon_schedule (Schedule): defines rate at which epsilon decays
      test_epsilon (float): epsilon to use during test time (when test is
        True in act)
      state_embedder_factory (Callable --> StateEmbedder): type of state
        embedder to use
      exploration: bool whether or not this is the exploration policy
      gamma (float): discount factor
    """
    super().__init__()
    self._Q = DuelingNetwork(num_actions, state_embedder_factory(), exploration)
    self._target_Q = DuelingNetwork(num_actions, state_embedder_factory(), exploration)
    self._num_actions = num_actions
    self._epsilon_schedule = epsilon_schedule
    self._test_epsilon = test_epsilon
    self._gamma = gamma
    self._reward_relabeler = None

    # Used for generating statistics about the policy
    # Average of max Q values
    self._max_q = collections.deque(maxlen=1000)
    self._min_q = collections.deque(maxlen=1000)
    self._losses = collections.defaultdict(lambda: collections.deque(maxlen=1000))

    self.exploration = exploration

  def act(self, state, z, prev_hidden_state=None, test=False):
    """
    Args:
      state (State)
      test (bool): if True, takes on the test epsilon value
      prev_hidden_state (object | None): unused agent state.
      epsilon (float | None): if not None, overrides the epsilon greedy
      schedule with this epsilon value. Mutually exclusive with test
      flag

    Returns:
      int: action
      hidden_state (None)
    """
    del prev_hidden_state

    q_values, hidden_state = self._Q([state], z, None)
    if test:
      epsilon = self._test_epsilon
    else:
      epsilon = self._epsilon_schedule.step()
    self._max_q.append(torch.max(q_values).item())
    self._min_q.append(torch.min(q_values).item())
    return epsilon_greedy(q_values, epsilon)[0], None

  def loss(self, experiences, weights, z):
    """Updates parameters from a batch of experiences

    Minimizing the loss:

      (target - Q(s, a, z))^2

      target = r if done
           r + \gamma * max_a' Q(s', a')

    Args:
      experiences (list[Experience]): batch of experiences, state and
        next_state may be LazyFrames or np.arrays
      weights (list[float]): importance weights on each experience

    Returns:
      loss (torch.tensor): MSE loss on the experiences.
    """
    batch_size = len(experiences)
    states = [e.state for e in experiences]
    actions = torch.tensor([e.action for e in experiences]).long()
    next_states = [e.next_state for e in experiences]
    rewards = torch.tensor([e.reward for e in experiences]).float()

    # (batch_size,) 1 if was not done, otherwise 0
    not_done_mask = torch.tensor([1 - e.done for e in experiences]).byte()
    weights = torch.tensor(weights).float()

    # TODO(evzliu): Could more gracefully incorporate aux_losses
    current_state_q_values, aux_losses = self._Q(states, z, None)
    if isinstance(aux_losses, dict):
      for name, loss in aux_losses.items():
        self._losses[name].append(loss.detach().cpu().data.numpy())
    current_state_q_values = current_state_q_values.gather(
        1, actions.unsqueeze(1))

    # DDQN
    best_actions = torch.max(self._Q(next_states, z, None)[0], 1)[1].unsqueeze(1)
    next_state_q_values = self._target_Q(next_states, z, None)[0].gather(
        1, best_actions).squeeze(1)
    targets = rewards + self._gamma * (
      next_state_q_values * not_done_mask.float())
    targets.detach_()  # Don't backprop through targets

    td_error = current_state_q_values.squeeze() - targets
    loss = torch.mean((td_error ** 2) * weights)
    self._losses["td_error"].append(loss.detach().cpu().data.numpy())
    aux_loss = 0
    if isinstance(aux_losses, dict):
      aux_loss = sum(aux_losses.values())
    return loss + aux_loss

  def sync_target(self):
    """Syncs the target Q values with the current Q values"""
    self._target_Q.load_state_dict(self._Q.state_dict())

  def set_reward_relabeler(self, reward_relabeler):
    """Sets the reward relabeler when computing the loss.

    Args:
      reward_relabeler (RewardLabeler)

    Raises:
      ValueError: when the reward relabeler has already been set.
    """
    if self._reward_relabeler is not None:
      raise ValueError("Reward relabeler already set.")
    self._reward_relabeler = reward_relabeler

  @property
  def stats(self):
    """See comments in constructor for more details about what these stats
    are"""
    def mean_with_default(l, default):
      if len(l) == 0:
        return default
      return np.mean(l)

    stats = {
        "epsilon": self._epsilon_schedule.step(take_step=False),
        "Max Q": mean_with_default(self._max_q, None),
        "Min Q": mean_with_default(self._min_q, None),
    }
    for name, losses in self._losses.items():
      stats[name] = np.mean(losses)
    return stats


class RecurrentDQNPolicy(DQNPolicy):
  """Implements a DQN policy that uses an RNN on the observations."""

  def loss(self, experiences, weights, z):
    """Updates recurrent parameters from a batch of sequential experiences

    Minimizing the DQN loss:

      (target - Q(s, a))^2

      target = r if done
           r + \gamma * max_a' Q(s', a')

    Args:
      experiences (list[list[Experience]]): batch of sequences of experiences.
      weights (list[float]): importance weights on each experience

    Returns:
      loss (torch.tensor): MSE loss on the experiences.
    """
    unpadded_experiences = experiences
    experiences, mask = utils.pad(experiences)
    batch_size = len(experiences)
    seq_len = len(experiences[0])

    hidden_states = [seq[0].agent_state for seq in experiences]
    # Include the next states in here to minimize calls to _Q
    states = [
        [e.state for e in seq] + [seq[-1].next_state] for seq in experiences]
    actions = torch.tensor(
        [e.action for seq in experiences for e in seq]).long()
    next_hidden_states = [seq[0].next_agent_state for seq in experiences]
    next_states = [[e.next_state for e in seq] for seq in experiences]
    rewards = torch.tensor(
        [e.reward for seq in experiences for e in seq]).float()

    # TODO(evzliu): Could more gracefully handle this by passing a
    # TrajectoryExperience object to label_rewards to take TrajectoryExperience
    # Relabel the rewards on the fly
    if self._reward_relabeler is not None:
      trajectories = [seq[0].trajectory for seq in experiences]
      # (batch_size, max_seq_len)
      indices = torch.tensor(
          [[e.index for e in seq] for seq in experiences]).long()

      # (batch_size * max_trajectory_len)
      rewards = self._reward_relabeler.label_rewards(
          trajectories)[0].gather(-1, indices).reshape(-1)

    # (batch_size,) 1 if was not done, otherwise 0
    not_done_mask = ~(torch.tensor(
        [e.done for seq in experiences for e in seq]).bool())
    weights = torch.tensor(weights).float()

    # (batch_size, seq_len + 1, actions)
    q_values, _ = self._Q(states, z, hidden_states)
    current_q_values = q_values[:, :-1, :]
    current_q_values = current_q_values.reshape(batch_size * seq_len, -1)
    # (batch_size * seq_len, 1)
    current_state_q_values = current_q_values.gather(1, actions.unsqueeze(1))

    # TODO(evzliu): Could more gracefully incorporate aux_losses
    aux_losses = {}
    if hasattr(self._Q._state_embedder, "aux_loss"):
      aux_losses = self._Q._state_embedder.aux_loss(unpadded_experiences)
      if isinstance(aux_losses, dict):
        for name, loss in aux_losses.items():
          self._losses[name].append(loss.detach().cpu().data.numpy())

    # DDQN
    next_q_values = q_values[:, 1:, :]
    # (batch_size * seq_len, actions)
    next_q_values = next_q_values.reshape(batch_size * seq_len, -1)
    best_actions = torch.max(next_q_values, 1)[1].unsqueeze(1)
    # Using the same hidden states for target
    target_q_values, _ = self._target_Q(next_states, z, next_hidden_states)
    target_q_values = target_q_values.reshape(batch_size * seq_len, -1)
    next_state_q_values = target_q_values.gather(1, best_actions).squeeze(1)
    targets = rewards + self._gamma * (
        next_state_q_values * not_done_mask.float())
    targets.detach_()  # Don't backprop through targets

    td_error = current_state_q_values.squeeze() - targets
    weights = weights.unsqueeze(1) * mask.float()
    loss = (td_error ** 2).reshape(batch_size, seq_len) * weights
    loss = loss.sum() / mask.sum()  # masked mean
    return loss + sum(aux_losses.values())

  def act(self, state, z, prev_hidden_state=None, test=False):
    """
    Args:
      state (State)
      test (bool): if True, takes on the test epsilon value
      prev_hidden_state (object | None): unused agent state.
      epsilon (float | None): if not None, overrides the epsilon greedy
      schedule with this epsilon value. Mutually exclusive with test
      flag

    Returns:
      int: action
      hidden_state (None)
    """
    q_values, hidden_state = self._Q([[state]], z, prev_hidden_state)
    if test:
      epsilon = self._test_epsilon
    else:
      epsilon = self._epsilon_schedule.step()
    self._max_q.append(torch.max(q_values).item())
    self._min_q.append(torch.min(q_values).item())
    return epsilon_greedy(q_values, epsilon)[0], hidden_state


class DQN(nn.Module):
  """Implements the Q-function."""
  def __init__(self, num_actions, state_embedder, exploration, latent_dim=8):
    """
    Args:
      num_actions (int): the number of possible actions at each state
      state_embedder (StateEmbedder): the state embedder to use
    """
    super(DQN, self).__init__()
    self._state_embedder = state_embedder
    self.exploration = exploration

    if self.exploration:
      self._q_values = nn.Linear(self._state_embedder.embed_dim + latent_dim, num_actions)
    else:
      self._q_values = nn.Linear(self._state_embedder.embed_dim, num_actions)

  def forward(self, states, z, hidden_states=None):
    """Returns Q-values for each of the states.

    Args:
      states (FloatTensor): shape (batch_size, 84, 84, 4)
      hidden_states (object | None): hidden state returned by previous call to
        forward. Must be called on constiguous states.

    Returns:
      FloatTensor: (batch_size, num_actions)
      hidden_state (object)
    """
    state_embed, hidden_state = self._state_embedder(states, hidden_states)
    
    # concatenate state and task encoding
    # if z != None:
    #   state_embed = torch.cat((state_embed, z.detach()), dim=0)
    # return self._q_values(state_embed), hidden_state
    if z != None:
      if state_embed.dim() == 2:
        state_embed = torch.cat((state_embed, z.detach()), dim=1)
      else:
        # tile z to same dims as state_embedding
        # state_embedding 32 x 11 x 64
        # z is 1 x 1 x 8 -> 32 x 11 x 8
        # concat: 32 x 11 x 72
        x = list(state_embed.shape)
        x[-1] = 1

        z = z.unsqueeze(0)
        z = z.repeat(tuple(x))
        
        state_embed = torch.cat((state_embed, z.detach()), dim=-1)

    return self._q_values(state_embed), hidden_state


class DuelingNetwork(DQN):
  """Implements the following Q-network:

    # Q(s, a) = V(s) + A(s, a) - avg_a' A(s, a')
    Q(s, a, z) = V(s, z) + A(s, a, z) - avg_a' A(s, a', z)
  """
  def __init__(self, num_actions, state_embedder, exploration, latent_dim=8):
    super(DuelingNetwork, self).__init__(num_actions, state_embedder, exploration)
    self.exploration = exploration

    # HERE breakpoint()
    print(self._state_embedder.embed_dim)

    if self.exploration:
      self._V = nn.Linear(self._state_embedder.embed_dim + latent_dim, 1)
      self._A = nn.Linear(self._state_embedder.embed_dim + latent_dim, num_actions)
    else:
      self._V = nn.Linear(self._state_embedder.embed_dim, 1)
      self._A = nn.Linear(self._state_embedder.embed_dim, num_actions)

  def forward(self, states, z, hidden_states=None):
    state_embedding, hidden_state = self._state_embedder(states, hidden_states)
    # breakpoint()
    # concatenate state and task encoding
    # print(f'state embedding {state_embedding.shape} z {z.shape}')

    if z != None:
      if state_embedding.dim() == 2:
        state_embedding = torch.cat((state_embedding, z.detach()), dim=1)
      else:
        # tile z to same dims as state_embedding
        # state_embedding 32 x 11 x 64
        # z is 1 x 1 x 8 -> 32 x 11 x 8
        # concat: 32 x 11 x 72
        x = list(state_embedding.shape)
        x[-1] = 1

        z = z.unsqueeze(0)
        z = z.repeat(tuple(x))

        state_embedding = torch.cat((state_embedding, z.detach()), dim=-1)

    V = self._V(state_embedding)
    advantage = self._A(state_embedding)
    mean_advantage = torch.mean(advantage)
    return V + advantage - mean_advantage, hidden_state


def epsilon_greedy(q_values, epsilon):
  """Returns the index of the highest q value with prob 1 - epsilon,
  otherwise uniformly at random with prob epsilon.

  Args:
    q_values (Variable[FloatTensor]): (batch_size, num_actions)
    epsilon (float)

  Returns:
    list[int]: actions
  """
  batch_size, num_actions = q_values.size()
  _, max_indices = torch.max(q_values, 1)
  max_indices = max_indices.cpu().data.numpy()
  actions = []
  for i in range(batch_size):
    if np.random.random() > epsilon:
      actions.append(max_indices[i])
    else:
      actions.append(np.random.randint(0, num_actions))
  return actions
