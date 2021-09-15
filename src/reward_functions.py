def simple(predicted_observation, _, info):
    x_pos, x_dot, predicted_theta, theta_dot = predicted_observation

    uncertainty = info["predicted_std"]
    reward = 3- ((1-predicted_theta-abs(x_pos *  0.1))*(1-uncertainty))

    return reward


def centered(predicted_observation, *args):
    """
    The more centered the better the reward
    :return:
    """
    x_pos, x_dot, predicted_theta, theta_dot = predicted_observation

    return 1-abs(x_pos)


def boundaries(predicted_observation, *args):
    x_pos, x_dot, predicted_theta, theta_dot = predicted_observation

    if abs(x_pos) > 2:
        return -100
    else:
        return 1