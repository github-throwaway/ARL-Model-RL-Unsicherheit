# ARL-Model-RL-Unsicherheit

## Setup

```
git clone https://github.com/github-throwaway/ARL-Model-RL-Unsicherheit.git
cd ARL-Model-RL-Unsicherheit/
pip install -r requirements.txt
```

## How to run
## Simple
Uses preconfigured system with trained model and agent with default configuration.
Simply run `main.py::main` to execute. 

### Advanced
You can also generate your own dataset and train your model with different datasets.
Please read code.

## Configuration
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



## Sources
- [Regression with probabilisitic layers](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html)
- [Modeling Epistemic and Aleatoric Uncertainty
with Bayesian Neural Networks and Latent Variables](https://mediatum.ub.tum.de/doc/1482483/1482483.pdf)

