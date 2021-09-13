# ARL-Model-RL-Unsicherheit

>This project was developed by Sophia Sigethy, Georg Staber, Simon Lund, and Malte Wilhelm for the [Applied Reinforcement Learning SS 21](https://www.dbs.ifi.lmu.de/cms/studium_lehre/lehre_master/parl21/index.html) course at LMU.

## :ledger: Index

- [Installation](#gear-installation)
- [How to run](#how-to-run)
  - [Simple](#slightly_smiling_face-simple)
  - [Advanced](#trophy-advanced)
- [Configuration](#hammer_and_wrench-configuration)
- [Sources](#books-sources)

## :gear: Installation
```
git clone https://github.com/github-throwaway/ARL-Model-RL-Unsicherheit.git
cd ARL-Model-RL-Unsicherheit/
pip install -r requirements.txt
```

## How to run
### :slightly_smiling_face: Simple
Uses preconfigured system with trained model and agent with[ default configuration](#hammer_and_wrench-configuration).
Simply run `main.py::main` to execute. 

### :trophy: Advanced
You can also generate your own dataset and train your model with different datasets. Please read code.

## :hammer_and_wrench: Configuration
The project was evvaluated using the following parameters.
```
1. Training Data Environment
noisysector = 0 - π
noise offset = 0.5
observation space = discrete
action space = 10 actions

2. Neural Network Settings
Epochs = 1000
time steps = 4

3. RL policy
reward function = []
RL algorithms = [PPO]
```



## :books: Sources
- [Regression with probabilisitic layers](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html)
- [Modeling Epistemic and Aleatoric Uncertainty
with Bayesian Neural Networks and Latent Variables](https://mediatum.ub.tum.de/doc/1482483/1482483.pdf)