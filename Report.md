
<br />
  <h3 align="center">Project 3 Collaboration and Competition</h3>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#multi_agent-ddpg">Multi-agent ddpg</a></li>
    <li><a href="#testing-maddpg">Testing MADDPG</a></li>
    <li><a href="#conclusion-and-future-improvements">Conclusion and future improvements</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

## About The Project
In this environment, each agent is racket that can make a ball bounce in a tennis field. If an agent hits the ball over 
the net of the field, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, 
it receives a reward of -0.01. The goal of each agent is to keep the ball in play.

It is not like tennis: the agents can only move back and forth (two actions) in a continuous space. The observation space 
consists of 24 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own local observation.

The task is episodic (ends when the ball is lost and no agent makes contact), and in order to solve the environment, the
agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 
Specifically, the rewards that each agent received  are added (without discounting), to get a score for each agent. 
This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


## Multi Agent DDPG

Multi Agent DDPG is based on DDPG (deep deterministic policy gradients (1)), which is an actor-critic method. 
In DDPG the actor (which uses a neural network as function approximator) decides the action, 
so it takes the current state as an input and returns continuous values, one for every action. 
It is called deterministic because given a state it will return always the same action, and that is why adding noise to the action 
helps with exploration. The critic is another network which estimates the discounted reward of the action given the state.
The critic takes both, action and state, and provides a number Q(s,a), called q-value. Since the critic uses the action given by the actor, actor and critic are coupled. 
  
MADDPG differs from DDPG in which the planning is centralized but the execution is decentralized. What does this means?
By centralized planning, each agent only has direct acces to local observations, but exists a big critic, 
an entity that is aware of all the environment and guide the agents on how to updated their policies.
Since the module has global information, it guides local policies towards the optimal solution for the entire system. 
However, during testing (execution time), the agent is going to act independently and the central module removed. 
The agent already learned how to act. 

In MADDPG every agent has an observation space in a continuous action space. Each agent has:
- An actor network feed only with its local information, is the same as the DDPG actor. Then, when the actor is trained, the gradient is updated based on the guidance of the critic (the q-values).
- A critic network to estimate q-values, which is centralized. It uses the actions of all agents.

### Parameters
* Learning rate - Actor: 1e-4
* Learning rate - Critic: 1e-3
* Batch Size: 512
* Buffer Size: 1e6
* Gamma: 0.99
* Tau: 1e-3 for actor and 1e-2 for Critic
* Iterations of learning per update step: 2
* Noise decay rate: 0.99999999999

* Architecture Actor :  statex256x256xaction
* Architecture Critic : state+action*num_agentsx256x256x1
* Noise: Theta - 0.15, Sigma - 0.1 but uniform distribution changed to a random normal distribution

The network was trained for 5000 episodes and the environment was solved in the episode 600, as the plot shows.

<figure>
<img src="images/magent.png" alt="drawing" style="width:400px;" caption="f"/>
<figcaption><i>Figure 1. Evolution of rewards (score) for the first 5000 episodes. The blue line is the average over the last 100 episodes, getting the maximum of the 2 agents.</i></figcaption>
 </figure>



## Testing MADDPG

### Using a global critic
Why do not use a unique joint critic? If all agents are the same that could be possible. Then, there is only one critic network, and many actor networks. 
This was tested in the tennis environment. At the beginning, the critic seemed to learn and guide the actor, but after 400 iterations it could not learn more than 0.1. 
What does it change from one big centralized critic, to having a  separate critic for each agent? The big centralized critic, has to deal with more information and with the input of multiple actors.
This can lead to a condition in which the actors trained at one point diverge so much, that the q-values are not converging properly in the critic. 
Given the unstable nature of DDPG (many factors, like noise, learning rates, tau, batch size or replay buffer experience selection), it would be possible at the end to find a set 
of conditions to make it work (at least theoretically), but it was proven to be very difficult, so the version with one critic per agent was used.

### Tau values
By logic, the critic should be a little bit more advanced in prediction than the actor (since it is guiding the actor). That should 
provide then more stability. The learning rate is higher for the critic so it speeds up convergence over the actor, 
which learns slowly.  In the same line, when the update is happening, the tau value is increased. 

### Batch normalization 
It has not been proven in literature to improve to add value to the learning, but seems to help. 

### Noise annealing
The initial parameters from the OUNoise (theta = 0.15 and sigma=1.0) were good, however, know how to anneal the noise was the difficult 
part. In other words, to know the decreasing rate for the noise to don't go down quickly at the beginning. The annealing was fixed 
to stop in 500.000 steps and only reduce the factor of decreasing every 2 episodes. 

## Conclusion and future improvements

MADDPG is capable to train in a continuous space, even a difficult one, given better performance than individual trainings with DDPG.
However, it inherits the flaws of DDPG, specifically, the instability and sensitivity to initial parameters, like the exploration noise or 
initial random seeds.

It was also observed that having a unique big joint critic, which would save memory between the agents, is more unstable than using a critic
per agent. 

 There are still certain things that could be tested with MADDPG:
- Gradient inverter: The main idea is, instead of clipping the gradient using a tanh function between -1 and 1, to use a
function that when the gradient exceeds, to change the sign but keeping the magnitude that guides the gradient.    
- Noise networks (3) in DDPG agent.The noise topic is kind of critical and it has been demonstrated that there are better
strategies than the OUNoise, like Noise networks, in which the weight of the networks is slightly perturbed with noise 
during training to facilitate exploration. 
- Use of two critics instead of one, and take the minimum q-value of both (2). That would help to avoid overestimation of the q-function and
get better guidance for the actors. 
- Replay buffer management. It has been shown that the way we select the experiences in the replay buffer makes a difference.
This could be here the case, since the noise is annealed over time. 
- Huber loss. In the original paper, instead of the quadratic error, the huber loss function is used. The huber loss is 
the L1 loss (difference of absolute values) with a parameter to decide how close we want to get from the quadratic function
to the linear function. Using a Huber loss could help, even if it has not been proved to really make a difference. 

Like all the experiments in RL, it is a matter of trial and error, and see what works better for the environment. 
In addition, other algorithms could be used, like PPO or 

## References
* (1) [Experience Selection in Deep Reinforcement Learning for Control](https://jmlr.org/papers/v19/17-131.html)
* (2) [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
* (3) [Noisy networks](https://arxiv.org/pdf/1706.10295.pdf)
