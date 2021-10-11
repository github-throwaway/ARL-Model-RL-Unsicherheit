# UNCERT: Semi-Model-Based RL with Uncertainty

>This project was developed by Simon Lund, Sophia Sigethy, Georg Staber, and Malte Wilhelm for the [Applied Reinforcement Learning SS 21](https://www.dbs.ifi.lmu.de/cms/studium_lehre/lehre_master/parl21/index.html) course at LMU.



<img src="docs/cartpole.svg " width="100%" alt="Cover image">

## :ledger: Index

- [Deliverables](#memo-deliverables)
  - [Videos](#video_camera-videos)
- [Installation](#gear-installation)
- [How to run](#how-to-run)
  - [Simple](#slightly_smiling_face-simple)
  - [Advanced](#trophy-advanced)
- [Configuration](#hammer_and_wrench-configuration)
- [Sources](#books-sources)

## :memo: Deliverables
As part of the course we created an extensive [report](docs/21_SS_arl_uncertainty_report.pdf) as well as a final [presentation](docs/21_SS_arl_uncertainty_presentation.pdf) of the project.

### :video_camera: Videos
The RL agent swings up using either side.

https://user-images.githubusercontent.com/25488086/135059144-6a8d22c5-b30b-4a9f-a6e2-3c40a838ee5b.mp4

The RL agent avoids the noisy section on the left and swings up on the right side.

https://user-images.githubusercontent.com/25488086/135059119-d990ded4-2e32-4744-8a3a-c4afc5bee633.mp4


## :gear: Installation

Start a development environment in your browser by clicking the button below. 

### Local Installtion

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](gitpod.io/#https://github.com/github-throwaway/ARL-Model-RL-Unsicherheit)


```bash
git clone https://github.com/github-throwaway/ARL-Model-RL-Unsicherheit.git
cd ARL-Model-RL-Unsicherheit/
pip install -r requirements.txt # or python setup.py install
```

## How to run
### :slightly_smiling_face: Simple
Uses preconfigured system with trained model and agent with [default configuration](#hammer_and_wrench-configuration). 
```bash
cd src/
python main.py
```

### :trophy: Advanced
For the sake of usability, we implemented an argument parser. By passing some predefined arguments to the python program call, it is possible to start different routines and also change hyperparameters needed by the algorithms. This enables the user to run multiple tests with different values without making alterations to the code. This is especially helpful when fine-tuning hyperparameters for reinforcement learning algorithms, like PPO. To get an overview of all the possible arguments, and how these arguments can be used, the user may call `python main.py --help`.

## :hammer_and_wrench: Configuration
The project was evaluated using the following parameters.
```
1. Training Data Environment
noisysector = 0 - π
noise offset = 0.5
observation space = discrete
action space = 10 actions

2. Neural Network Settings
Epochs = 100
time steps = 4

3. RL policy
reward function = [simple,centered,right,boundaries,best,cos,xpos_theta_uncert]
RL algorithms = PPO
```



## :books: Sources
- [Regression with probabilisitic layers](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html)
- [Modeling Epistemic and Aleatoric Uncertainty
with Bayesian Neural Networks and Latent Variables](https://mediatum.ub.tum.de/doc/1482483/1482483.pdf)
- [BLiTZ — A Bayesian Neural Network library for PyTorch](https://towardsdatascience.com/blitz-a-bayesian-neural-network-library-for-pytorch-82f9998916c7)
