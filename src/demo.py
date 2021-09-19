import math

import agents
import evaluation
import usuc
import utils


def demo_noisy_env():
    print("=== Noisy Environment ===",
          "These examples are only to demonstrate the use of the library and its functions, "
          "and the trained agents may not solve the environments.",
          "- 1. Environment: Left half is noisy =>  Agent tries to swing pole upwards preferably via the right half.",
          "- 2. Environment: Right half is noisy => Agent tries to swing pole upwards preferably via the left half.",
          "", sep="\n")
    # env where left where left half is noisy
    print("training agent on first environment...")
    env = usuc.USUCEnv(noise_offset=0.3, noisy_circular_sector=(0, math.pi))
    ppo = agents.create("ppo", env)
    agents.train(ppo, total_timesteps=80000)
    agents.run(ppo, env, 10)
    env.close()

    # env where right where left half is noisy
    print("training agent on second environment...")
    env = usuc.USUCEnv(noise_offset=0.3, noisy_circular_sector=(math.pi, 2 * math.pi))
    ppo = agents.create("ppo", env)
    agents.train(ppo, total_timesteps=80000)
    agents.run(ppo, env, 10)
    env.close()


def demo_usuc_pole_rotation():
    """
    Demonstrate the correlation between the angular progression and the pole rotation
    """

    print("=== Angular Progression ===")
    print("Rotates the pole starting from 0° to 360°")

    env = usuc.USUCEnv(noisy_circular_sector=(0, 0), noise_offset=0, render=True)
    for theta in range(0, 360):
        env.reset(math.radians(theta), 0)
        env.render()

        if theta % 90 == 0:
            print(f"{theta}° <=> {theta / 180} * pi")

    env.close()


def demo_usuc_random_actions():
    """
    Demonstrate the USUC Env with random actions
    """

    # init with random angle and noisy circular sector
    noisy_circular_sector = (0, math.pi)
    noise_offset = 0.3
    random_angle = utils.random_start_theta()

    env = usuc.USUCDiscreteEnv(
        num_actions=10,
        noisy_circular_sector=noisy_circular_sector,
        noise_offset=noise_offset,
        render=True,
    )
    env.reset(random_angle)

    print("=== Run with Random Actions ===")
    print("Initialization with:")
    print("- Noisy Circular Sector:", noisy_circular_sector)
    print("- Noise Offset:", noise_offset)
    print("- Random Angle:", random_angle)

    history = utils.random_actions(env)
    env.close()

    # plot data
    evaluation.plot_angles(history, "no model")


def demo():
    demo_usuc_pole_rotation()
    demo_usuc_random_actions()
    demo_noisy_env()


if __name__ == "__main__":
    demo()
