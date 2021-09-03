def simple(predicted_observation, reward, info, action):
    x_pos, x_dot, predicted_theta, theta_dot = predicted_observation
    # reward = cos(theta)
    x_pos_weight = 1
    uncertainty = info["predicted_std"]
    reward = reward * (1-uncertainty) - abs(x_pos * x_pos_weight)

    return reward


def centered(predicted_observation, reward, info, action):
    """
    The more centered the better the reward
    :return:
    """
    x_pos, x_dot, predicted_theta, theta_dot = predicted_observation

    return 1-abs(x_pos)


def boundaries(predicted_observation, reward, info, action):
    x_pos, x_dot, predicted_theta, theta_dot = predicted_observation

    if abs(x_pos) > 2:
        return -100
    else:
        return 1