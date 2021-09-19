from stable_baselines3 import PPO, A2C
from stable_baselines3.common import callbacks
from tqdm import tqdm


class PBarCallback(callbacks.BaseCallback):
    def __init__(self, verbose=0):
        super(PBarCallback, self).__init__(verbose)
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.locals['total_timesteps'])

    def _on_step(self):
        self.pbar.n = self.n_calls
        self.pbar.update(0)
        return True

    def _on_training_end(self):
        self.pbar.n = self.n_calls
        self.pbar.update(0)
        self.pbar.close()


def save(agent, filepath) -> None:
    """
    Saves the agent

    :param agent: Stable_baselines3 agent
    :param filepath: Filepath where agent is stored
    """
    agent.save(filepath)

def load(agent:str, filepath):
    """
    Loads a saved agent
    :param agent: The algorithm used to train the agent
    :param filepath: The path where the agent is stored
    :return: The loaded agent
    """
    if agent == "ppo":
        return PPO.load(filepath)
    elif agent == "a2c":
        return A2C.load(filepath)
    else:
        raise NotImplementedError(f"agent '{agent}' unknown")

def create(agent: str, env):
    """
    Returns untrained, instantiated stable_baselines3 agent

    :param agent: "ppo" or "a2c"
    :param env: Instantiated env
    """
    if agent == "ppo":
        return PPO("MlpPolicy", env, verbose=0)
    elif agent == "a2c":
        return A2C("MlpPolicy", env, verbose=0)
    else:
        raise NotImplementedError(f"agent '{agent}' unknown")


def train(agent, total_timesteps) -> None:
    """
    Trains the agent

    :param agent: Stable_baselines3 agent
    :param total_timesteps: Number of timesteps the agent is trained
    """

    cb = callbacks.CallbackList([
        PBarCallback(total_timesteps)
    ])

    agent.learn(total_timesteps, cb)


def run(agent, env, runs):
    """
    Runs the agent on the given env

    :param agent: Stable_baselines3 agent
    :param env: Instantiated env
    :param runs: Nunber of runs with the agent on the env
    :return: History for each run
    """
    histories = []

    for _ in range(runs):
        obs = env.reset()

        history = []

        for i in tqdm(range(1000)):
            action, _states = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            history.append((obs, reward, done, info))
            env.render()

            if done:
                break

        histories.append(history)

    return histories
