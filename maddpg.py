import random
import copy
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import model
from agent import Agent
from model import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

UPDATE_EVERY = 10
BATCH_SIZE = 256
LEARN_N = 2
LR_CRITIC = 1e-3        # learning rate of the critic
GAMMA = 0.99

class MADDPG():
    """Multi agent class for training that contains the DDPG agents and shared replay buffer."""

    def __init__(self, state_size = 8,action_size=2, seed=0,
                 num_agents=2,
                 buffer_size=1e6,
                 batch_size=256,
                 gamma=0.99,
                 update_every=10,
                 noise_start=1.0,
                 noise_decay=0.999,
                 t_stop_noise=30000):
        """
        Params
        ======
            action_size (int): dimension of each action
            seed (int): Random seed
            n_agents (int): number of distinct agents
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            noise_start (float): initial noise weighting factor
            noise_decay (float): noise decay rate
            update_every (int): how often to update the network
            t_stop_noise (int): max number of timesteps with noise applied in training
        """

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.num_agents = num_agents
        self.noise_eps = noise_start
        self.noise_decay = noise_decay
        self.t_step = 0
        self.noise_on = True
        self.t_stop_noise = t_stop_noise
        self.action_size = action_size
        self.state_size = state_size

        # create N agents
        self.agents = [Agent(state_size=self.state_size, action_size=self.action_size, agent_id = id, seed=7) for id in range(num_agents)]
        # create shared replay buffer
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        # declare joint critic for all agents
        self.critic_local = Critic(num_agents, state_size, action_size, seed).to(device)
        self.critic_target = Critic(num_agents, state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        Agent.complete_update(self.critic_local, self.critic_target)

    def act(self, states):
        # pass each agent's state from the environment and calculate its action
        joint_actions = []
        for agent, state in zip(self.agents, states):
            action = agent.act(state, )
            self.noise_weight *= self.noise_decay
            joint_actions.append(action)
        return np.array(joint_actions).reshape(1, -1)  # reshape 2x2 into 1x4 dim vector

    def step(self, states, actions, rewards, next_states, dones):
            """Save experience in replay memory, and use random sample from buffer to learn."""
            states = states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
            next_states = next_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
            self.memory.add(states, actions, rewards, next_states, dones)

            # update time steps
            if self.t_step > self.t_stop_noise:
                self.noise_eps = 0

            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                # Learn, if enough samples are available in memory
                if len(self.memory) > BATCH_SIZE:
                    for _ in range(LEARN_N):
                        experiences = self.memory.sample()
                        self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # each agent uses its own actor to calculate next_actions
        all_next_actions = []
        all_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            # extract agent i's state and get action via actor network
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            all_actions.append(action)
            # extract agent i's next state and get action via target actor network
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            all_next_actions.append(next_action)

        # ---------------------------- update critic ---------------------------- #
        # get predicted next-state actions and Q values from target models
        self.critic_optimizer.zero_grad()
        agent_id = torch.tensor([agent_id]).to(device)
        actions_next = torch.cat(all_next_actions, dim=1).to(device)
        with torch.no_grad():
            q_targets_next = self.critic_target(next_states, actions_next)
        # compute Q targets for current states (y_i)
        q_expected = self.critic_local(states, actions)
        # q_targets = reward of this timestep + discount * Q(st+1,at+1) from target network
        q_targets = rewards.index_select(1, agent_id) + (
                gamma * q_targets_next * (1 - dones.index_select(1, agent_id)))
        # compute critic loss
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        # minimize loss
        critic_loss.backward()
        self.critic_optimizer.step()
        # ----------------------- update critic target network ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)


        # each agent learns from its experience sample
        for i, agent in enumerate(self.agents):
            # ---------------------------- update actor ---------------------------- #
            # compute actor loss
            self.agent.actor_optimizer.zero_grad()
            # detach actions from other agents
            actions_pred = [actions if i == self.id else actions.detach() for i, actions in enumerate(all_actions)]
            actions_pred = torch.cat(actions_pred, dim=1).to(device)
            actor_loss = -self.critic_local(states, actions_pred).mean()
            # minimize loss
            actor_loss.backward()
            self.agent.actor_optimizer.step()


            self.soft_update(self.actor_local, self.actor_target, self.tau)



    def save_agents(self):
        # save models for local actor and critic of each agent
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(),  f"checkpoint_actor_agent_{i}.pth")
            torch.save(agent.critic_local.state_dict(), f"checkpoint_critic_agent_{i}.pth")


class DDPG():
    """DDPG agent with own actor and critic."""

    def __init__(self, agent_id, model, action_size=2, seed=0,
                 tau=1e-3,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 weight_decay=0.0):
        """
        Params
        ======
            model: model object
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
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
       
        # Actor Network
        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic Network
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)
        
        # Set weights for local and target actor, respectively, critic the same
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)
        
        # Noise process
        self.noise = OUNoise(action_size, seed)






    def soft_update(self, local_model, target_model, tau):
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

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
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
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
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