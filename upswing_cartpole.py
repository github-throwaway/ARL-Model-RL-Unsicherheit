import math
import numpy as np
from random import uniform
from gym.envs.classic_control.cartpole import CartPoleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym import spaces, logger


class UpswingCartPoleEnv(CartPoleEnv):
    """Customized cartpole environment
    TODO: eventually replace with env mentioned by mr. tokic
    """

    def __init__(self, offset=2, min_range=-10, max_range=10):
        super().__init__()
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)


        self.theta_threshold_radians = math.inf
        self.min_range = min_range
        self.max_range = max_range
        self.offset = offset

    def pre_step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot, fake_theta = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot,fake_theta)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def step(self, action):
        observation, reward, done, info = self.pre_step(action)
        pole_angle = self.state[2]
        # todo: should we also implement uncertainty for a region of the
        #  track?

        if pole_angle > self.min_range and pole_angle < self.max_range:
            fake_observation = pole_angle + uniform(-self.offset,self.offset)
            observation[4] = fake_observation
            info["uncertain"] = True
        else:
            observation[4] = pole_angle
            info["uncertain"] = False

        return observation, reward, done, info

    def reset(self):
        """This cart pole should start facing downwards. So we need to add pi to the initial pole position (rad) """
        super().reset()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(5,))
        self.state[2] += math.pi
        self.state[4] += math.pi

        return np.array(self.state)


if __name__ == "__main__":
    env = UpswingCartPoleEnv()
    env = make_vec_env(type(env), n_envs=4)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2500)
    model.save("save_model")
    del model  # remove to demonstrate saving and loading
    model = PPO.load("save_model")

    observation = env.reset()
    for _ in range(1000):
        env.render()
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)
        print(f"original: {observation[1,2]} - fake: {observation[1,4]} - "
              f"diff: {observation[1,2]-observation[1,4]}")
        print()

        if done[1]:
            break

    env.close()
