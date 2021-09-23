from setuptools import setup

setup(
    name="UNCERT",
    version="1.0.0",
    install_requires=[
        "gym~=0.18.3",
        "pandas==1.2.5",
        "matplotlib~=3.4.3",
        "numpy~=1.19.5",
        "stable-baselines3",
        "gym-cartpole-swingup",
        "dill~=0.3.4",
        "tqdm~=4.62.2",
        "gpytorch~=1.5.1",
        "torch",
        "scikit-learn~=0.24.2",
        "blitz-bayesian-pytorch",
        "tikzplotlib",
    ],
    url="https://github.com/github-throwaway/ARL-Model-RL-Unsicherheit",
    author="Simon Lund, Sophia Sigethy, Georg Staber, and Malte Wilhelm",
    description="A probabilistic neural network for modeling the system "
    "dynamics of the Cartpole environment in OpenAI gym. This "
    "network emulates the environment to train a Reinforcement "
    "Learning policy. ",
)
