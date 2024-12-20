import gym
import d4rl  # Make sure d4rl is installed and properly configured
import numpy as np

def inspect_environment(env_name, num_steps=5):
    """
    Inspect the environment inputs and updates.

    Args:
        env_name (str): The name of the gym environment to load.
        num_steps (int): Number of steps to simulate in the environment.
    """
    try:
        # Load the environment
        env = gym.make(env_name)
        print(f"Environment '{env_name}' loaded successfully!")

        # Reset the environment to get the initial observation
        initial_obs = env.reset()
        print("\nInitial Observation:")
        print(initial_obs)

        # Iterate through a few steps in the environment
        print("\nSimulating steps...")
        for step in range(num_steps):
            # Sample a random action from the environment's action space
            action = env.action_space.sample()
            print(f"\nStep {step + 1}:")
            print(f"Action: {action}")

            # Take a step in the environment
            next_obs, reward, done, info = env.step(action)

            # Display outputs from the environment
            print(f"Next Observation: {next_obs}")
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            print(f"Info: {info}")

            # Break if the environment signals termination
            if done:
                print("\nEnvironment signaled termination. Resetting...")
                initial_obs = env.reset()
                print("Environment reset!")
    except gym.error.UnregisteredEnv:
        print(f"Environment '{env_name}' is not registered.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    env_name = "hopper-medium-v0"  # Replace with your desired environment
    inspect_environment(env_name)
