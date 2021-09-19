import math


def simple(predicted_observation, _, info):
    x_pos, x_dot, predicted_theta_sin, predicted_theta_cos, theta_dot = predicted_observation

    uncertainty = info["predicted_std"]
    reward = 3 - ((1-predicted_theta_cos-abs(x_pos *  0.1))*(1-uncertainty))

    return reward


def centered(obs, reward, info):
    """
    The more centered the better the reward
    :return:
    """
    x_pos = obs.x_pos

    x_pos_rew = (max(abs(x_pos) - 1, 0)) ** 2

    return 1-x_pos_rew


def right(obs,reward, info):
    if obs.x_pos < 0:
        return -1
    else:
        return 1


def boundaries(obs, reward, info):
    if abs(obs.x_pos) > 2:
        return -100
    else:
        return 1



def best(obs, reward, info):
    x_pos = obs.x_pos

    x_pos_rew = (max(abs(x_pos) - 1, 0) )**2
    print(x_pos_rew)

    theta_cos = obs.theta_cos
    theta_sin = obs.theta_sin

    std_sin = info["predicted_std_sin"]
    std_cos = info["predicted_std_cos"]

    uncertainty = math.sqrt(std_sin**2 + std_cos**2)

    reward = theta_cos * (1 - 3 * uncertainty) - x_pos_rew

    return reward


def cos(obs, reward, info):
    return obs.theta_cos


