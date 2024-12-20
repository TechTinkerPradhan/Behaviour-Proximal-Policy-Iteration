import gym
import torch
from bppo import BehaviorProximalPolicyOptimization
import os
os.environ["MUJOCO_GL"] = "egl"


# Load the environment
env = gym.make('Hopper-v2')  # Use the standard gym Hopper environment
env.reset()

# Define device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained BPPO model
bppo_model_path = "logs/hopper-medium-v2/8/2024_12_03__01_22_56/bppo_best.pt"  
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Initialize the BPPO model with the same architecture as training
bppo = BehaviorProximalPolicyOptimization(
    device=device,
    state_dim=state_dim,
    hidden_dim=1024,  # Same as args.bppo_hidden_dim during training
    depth=2,          # Same as args.bppo_depth
    action_dim=action_dim,
    policy_lr=1e-4,   # Corrected from 'lr' to 'policy_lr'
    clip_ratio=0.25,  # Clip ratio used during training
    entropy_weight=0.0,
    decay=0.96,
    omega=0.9,
    batch_size=512    # Batch size (not used during inference)
)

# Load the saved model weights
bppo.load(bppo_model_path)
print("BPPO model loaded successfully!")

# Run the Hopper environment with the trained policy
state = env.reset()
for _ in range(1000):  # Number of timesteps
    env.render()  # Render the environment
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    action = bppo.act(state_tensor, is_sample=False).cpu().detach().numpy()  # Generate action
    state, reward, done, _ = env.step(action)  # Take action in the environment
    if done:
        state = env.reset()  # Reset if episode ends

env.close()
