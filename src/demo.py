import math
from src import usuc, data


def demo_usuc_pole_rotation():
    """
    Demonstrate the correlation between the angular progression and the pole rotation
    """

    print("=== Angular Progression ===")
    print("Rotates the pole starting from 0° to 360°")

    env = usuc.USUCEnv(render=True)
    for angle in range(0, 360):
        env.reset(math.radians(angle), 0)
        env.render()

        if angle % 90 == 0:
            print(f"{angle}° <=> {angle / 180} * pi")

    env.close()


def demo_usuc_random_actions():
    """
    Demonstrate the USUC Env with random actions
    """

    # init with random angle and noisy circular sector
    noisy_circular_sector = (0, math.pi)
    noise_offset = 0.1
    random_angle = usuc.random_start_angle()

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
    original_angles = [info["original_angle"] for (_, info) in history]
    observed_angles = [obs["theta"] for (obs, _) in history]
    data.plot_angles(original_angles, observed_angles)


if __name__ == "__main__":
    demo_usuc_pole_rotation()
    demo_usuc_random_actions()
