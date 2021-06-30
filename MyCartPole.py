import math
import random

from gym.envs.classic_control.cartpole import CartPoleEnv


env = CartPoleEnv()
env.theta_threshold_radians = 90 * 2 * math.pi / 360


observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  print(observation)

  degrees = math.degrees(observation[3])
  observation_ = observation

  if degrees > 10 and degrees < 30 :
      print(f"Pre:{degrees}")
      offset = random.randrange(0,1)
      observation_[3] = math.radians(degrees*offset)
      print(f"Post:{observation_[3]}")

  if done:
    observation = env.reset()
env.close()