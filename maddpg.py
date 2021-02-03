import random
import copy
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Critic, Actor

LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
TAU = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    """Multi agent designed for training. It contains the critics."""

    def __init__(self, num_agents = 2, state_size = 24, action_size=2,
                 buffer_size=100000,
                 batch_size=512,
                 gamma=0.99,
                 update_every=2,
                 noise_start=1.0,
                 noise_decay=0.99999,
                 stop_noise=50000, seed = 31):
        """
        Params
        ======
            state_size(int): dimension of each observation state
            action_size (int): dimension of each action
            n_agents (int): number of distinct agents
            buffer_size (int): replay buffer size
            batch_size (int)
            gamma (float): discount factor
            noise_start (float): initial noise weighting factor
            noise_decay (float): noise decay rate
            update_every (int): how often to update the target network
            stop_noise (int): max number of timesteps with noise applied in training
        """

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.num_agents = num_agents
        self.state_size = state_size
        self.noise_factor = noise_start
        self.noise_decay = noise_decay
        self.t_step = 0

        self.stop_noise = stop_noise

        # create two agents, each with their own actor and critic

        self.critic_local = [Critic(num_agents, state_size, action_size, seed).to(device) for _ in range(num_agents)]
        self.critic_target = [Critic(num_agents, state_size, action_size, seed).to(device) for _ in range(num_agents)]
        self.critic_optimizer = [ optim.Adam(self.critic_local[i].parameters(), lr=LR_CRITIC) for i in range(num_agents) ]

        self.agents = [Agent(i) for i in range(num_agents)]

        for i in range(self.num_agents):
            Agent.hard_copy_weights(self.critic_target[i], self.critic_local[i])

        # create shared replay buffer
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size)

    def step(self, n_states, n_actions, n_rewards, n_next_states, n_dones):
        n_states = n_states.reshape(1, -1)  # reshape into 1x48 for easier network input
        n_next_states = n_next_states.reshape(1, -1)
        self.memory.add(n_states, n_actions, n_rewards, n_next_states, n_dones)
        
        # if stop_noise time steps are achieved turn off noise
        if self.t_step > self.stop_noise:
            self.noise_decay = 1.0
            self.noise_factor = 1.0
        
        self.t_step = self.t_step + 1     
        # Learn every update_every time steps.
        if self.t_step % self.update_every == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                # sample from the replay buffer for each agent
                experiences = [self.memory.sample() for _ in range(self.num_agents)]
                self.learn(experiences, self.gamma)

    def act(self, n_states, add_noise=True):
        # calculate each action
        joint_actions = []
        for agent, state in zip(self.agents, n_states):
            action = agent.act(state, noise_weight=self.noise_factor, add_noise=add_noise)
            if add_noise:
                self.noise_factor *= self.noise_decay
            joint_actions.append(action)
        return np.array(joint_actions).reshape(1, -1)

    def learn(self, experiences, gamma):
        # each agent uses its own actor to calculate next_actions
        joint_next_actions = []
        joint_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            state = states.reshape(-1, self.num_agents, self.state_size).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            joint_actions.append(action)
            next_state = next_states.reshape(-1, self.num_agents, self.state_size).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            joint_next_actions.append(next_action)
                       
        # each agent learns from its experience sample
        for i, agent in enumerate(self.agents):
            states, actions, rewards, next_states, dones = experiences[i]
            # ---------------------------- update critic ---------------------------- #
            # get predicted next-state actions and Q values from target models
            self.critic_optimizer[i].zero_grad()
            agent_id = torch.tensor([i]).to(device)
            actions_next = torch.cat(joint_next_actions, dim=1).to(device)
            with torch.no_grad():
                next_state = next_states.reshape(-1, self.num_agents, self.state_size).index_select(1, agent_id).squeeze(1).to(device)
                q_targets_next = self.critic_target[i](next_state, actions_next)
            # compute Q targets for current states (y_i)
            state = states.reshape(-1, self.num_agents, self.state_size).index_select(1, agent_id).squeeze(1)
            q_expected = self.critic_local[i](state, actions)
            # q_targets = reward of this timestep + discount * Q(st+1,at+1) from target network
            q_targets = rewards.index_select(1, agent_id) + (
                        gamma * q_targets_next * (1 - dones.index_select(1, agent_id)))
            # compute critic loss
            critic_loss = F.mse_loss(q_expected, q_targets.detach())
            # minimize loss
            critic_loss.backward()
            self.critic_optimizer[i].step()

            # ---------------------------- update actor ---------------------------- #
            # compute actor loss
            agent.actor_optimizer.zero_grad()
            actions_pred = [actions if i == j else actions.detach() for j, actions in enumerate(joint_actions)]
            actions_pred = torch.cat(actions_pred, dim=1).to(device)
            state = states.reshape(-1, self.num_agents, self.state_size).index_select(1, agent_id).squeeze(1)
            actor_loss = -self.critic_local[i](state, actions_pred).mean()
            # minimize loss
            actor_loss.backward()
            agent.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            Agent.soft_update(self.critic_local[i], self.critic_target[i], TAU * 10)
            Agent.soft_update(agent.actor_local, agent.actor_target, TAU)

            
    def save(self):
        # save models for local actor and critic of each agent
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(),  f"checkpoint_actor_agent_{i}.pth")
            torch.save(self.critic_local[i].state_dict(), f"checkpoint_critic_agent_{i}.pth")


class Agent():
    """DDPG agent with just an actor."""
    def __init__(self, agent_id, state_size = 24, action_size=2, seed=0):
        """
        Params
        ======
            action_size (int): dimension of each action
            seed (int): Random seed
            tau (float): for soft update of target parameters
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            weight_decay (float): L2 weight decay
        """
        random.seed(seed)
        self.id = agent_id
        self.action_size = action_size
        self.state_size = state_size

        # Actor Network
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        Agent.hard_copy_weights(self.actor_target, self.actor_local)

        # Noise process for the act moment
        self.noise = OUNoise(action_size, seed)


    @staticmethod
    def hard_copy_weights(target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def act(self, state, noise_weight=1.0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        # calculate action values
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state.unsqueeze(0)).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            self.noise_val = self.noise.sample() * noise_weight
            action += self.noise_val
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        random.seed(seed)
        np.random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed = 47):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): Random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)