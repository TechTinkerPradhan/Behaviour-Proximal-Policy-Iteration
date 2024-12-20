# import gym
# import d4rl

# env = gym.make('hopper-medium-v2')  # Load the environment
# print(env)  # Print environment details

# #verigy the environment
# import gym
# import d4rl

# env = gym.make("hopper-medium-v2")
# print("Environment created:", env)

# # Example usage
# obs = env.reset()
# print("Initial observation:", obs)

# action = env.action_space.sample()
# print("Sampled action:", action)

# next_obs, reward, done, info = env.step(action)
# print("Next observation:", next_obs)
# print("Reward:", reward)
# print("Done:", done)
# print("Info:", info)

# done = False
# while not done:
#     action = env.action_space.sample()  # Sample random action
#     obs, reward, done, info = env.step(action)
#     print(f"Reward: {reward}, Done: {done}")
#Train a Policy: If this is part of a reinforcement learning project, replace the random actions with a policy trained using algorithms like PPO, 
# DDPG, or SAC.
import gym

print("Available Gym Environments:")
available_envs = [env_spec.id for env_spec in gym.envs.registry.all()]
for env in sorted(available_envs):
    print(env)

try:
    import d4rl
    print("\nD4RL Environments:")
    d4rl_envs = [env_spec.id for env_spec in gym.envs.registry.all() if 'd4rl' in env_spec.entry_point]
    for env in sorted(d4rl_envs):
        print(env)
except ImportError:
    print("\nD4RL is not installed or not properly imported.")







