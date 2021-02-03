[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

This project reproduces the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Clone the repo
   ```sh
   git clone https://github.com/josemiserra/tennis_maddpg
   ```
2. If you don't have Anaconda or Miniconda installed, go to [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install Miniconda in your computer (miniconda is a lightweight version of the Anaconda python environment). 

3. It is recommended that you install your own environment with Conda. Follow the instructions here: [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). After that, open an anaconda command prompt or a prompt, and activate your environment.
  ```sh
  activate your-environment
  ```
4. Install the packages present in requirements.txt
   ```sh
   pip install requirements.txt
   pip install mlagents
   ```
5. If you want to use pytorch with CUDA, it is recommmended to go to https://pytorch.org/get-started/locally/ and install pytorch following the instructions there, according to your CUDA installation.

6. Move into the folder of the project, and run jupyter notebook.
   ```sh
   jupyter notebook
   ```
   Alternatively you can execute from the python console using the execute_train.py for training the network.
   ```sh
    python execute_train.py
   ```

Additionally you can download the environment from one of the links below if your OS is different from Win64:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
   
   
### Instructions

Follow the instructions in `Tennis.ipynb` to get started with training. 
For more info about the algorithm and tests done, read the file Report.md.

## License

Distributed under the MIT License from Udacity Nanodegree. See `LICENSE` file for more information.


## Contact

Jose Miguel Serra Lleti - serrajosemi@gmail.com

Project Link: https://github.com/josemiserra/tennis_maddpg

