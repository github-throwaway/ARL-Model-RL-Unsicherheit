import math

import usuc
import evaluation


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
    noise_offset = 0.1
    random_angle = usuc.random_start_theta()

    env = usuc.USUCEnv(
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

    history = usuc.random_actions(env)
    env.close()

    # plot data
    evaluation.plot_angles(history)


if __name__ == "__main__":
    demo_usuc_pole_rotation()
    demo_usuc_random_actions()
