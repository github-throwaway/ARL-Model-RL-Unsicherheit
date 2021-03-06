import math


def centered(obs, reward, info):
    """
    The more centered the better the reward
    :return:
    """
    x_pos = obs.x_pos

    x_pos_rew = (max(abs(x_pos) - 1, 0)) ** 2

    return 1 - x_pos_rew


def right(obs, reward, info):
    if obs.x_pos < 0:
        return -1
    else:
        return 1


def boundaries(obs, reward, info):
    if abs(obs.x_pos) > 2:
        return -100
    else:
        return 1


def xpos_theta_uncert(obs, reward, info):
    x_pos = obs.x_pos

    x_pos_rew = (max(abs(x_pos) - 1, 0)) ** 2

    theta_cos = obs.theta_cos
    theta_sin = obs.theta_sin

    std_sin = info["predicted_std_sin"]
    std_cos = info["predicted_std_cos"]

    uncertainty = math.sqrt(std_sin ** 2 + std_cos ** 2)

    reward = theta_cos * (1 - 3 * uncertainty) - x_pos_rew

    return reward


def cos(obs, reward, info):
    return obs.theta_cos


def best(obs, reward, info):
    """
    Subtract a weakened uncertainty from cosine and use this as reward
    """
    std_sin = info["predicted_std_sin"]
    std_cos = info["predicted_std_cos"]
    uncertainty = math.sqrt(std_sin ** 2 + std_cos ** 2)
    return obs.theta_cos - (uncertainty * 0.1)
