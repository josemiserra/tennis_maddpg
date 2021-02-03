from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from maddpg import MADDPG


def trainFunction(state_size, action_size, n_episodes=4000, num_agents = 2):
    magent = MADDPG(action_size= action_size ,noise_start=1.0, seed=2,gamma=0.99, t_stop_noise=30000)
    scores = []
    scores_deque = deque(maxlen=100)
    scores_avg = []

    for i_episode in range(1, n_episodes + 1):
        rewards = []
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        if i_episode%2:
            update = True
        # loop over steps
        while True:
            # select an action
            joint_actions = magent.act(states, update)
            update = False
            # take action in environment and set parameters to new values
            env_info = env.step(joint_actions)[brain_name]
            next_states = env_info.vector_observations
            rewards_v = env_info.rewards
            done_v = env_info.local_done
            # update and train agent with returned information
            magent.step(states, joint_actions, rewards_v, next_states, done_v)
            states = next_states
            rewards.append(rewards_v)
            if any(done_v):
                break

        # calculate episode reward as maximum of individually collected rewards of agents
        episode_reward = np.max(np.sum(np.array(rewards), axis=0))

        scores.append(episode_reward)  # save most recent score to overall score array
        scores_deque.append(episode_reward)  # save most recent score to running window of 100 last scores
        current_avg_score = np.mean(scores_deque)
        scores_avg.append(current_avg_score)  # save average of last 100 scores to average score array

        print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, current_avg_score), end="")

        # log average score every 200 episodes
        if i_episode % 200 == 0:
            print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, current_avg_score))

        # break and report success if environment is solved
        if np.mean(scores_deque) >= .5 and i_episode%200==0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode,
                                                                                         np.mean(scores_deque)))
            magent.save()

if __name__ == "__main__":
    env = UnityEnvironment(file_name="Tennis.exe")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)

    print('States have length:', state_size)

    n_episodes = 3000
    scores = trainFunction(state_size, action_size, n_episodes)
    print(scores)


