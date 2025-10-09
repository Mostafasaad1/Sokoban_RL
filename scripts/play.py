import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import warnings
from sokoban_env import SokobanEnv

# Suppress pkg_resources deprecation warning from pygame (optional)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, backbone_hidden=256, head_hidden=128):
        super(ActorCritic, self).__init__()
        
        # Shared backbone: state_dim → 256 → 256
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, backbone_hidden),
            nn.ReLU(),
            nn.Linear(backbone_hidden, backbone_hidden),
            nn.ReLU(),
        )
        
        # Policy head: 256 → 128 → action_dim
        self.policy_head = nn.Sequential(
            nn.Linear(backbone_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, action_dim)
        )
        
        # Value head: 256 → 128 → 1
        self.value_head = nn.Sequential(
            nn.Linear(backbone_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1)
        )

    def forward(self, x):
        features = self.shared_net(x)
        return self.policy_head(features)

def preprocess_obs(obs):
    obs_flat = obs.flatten()
    obs_normalized = obs_flat.astype(np.float32) / 6.0
    return torch.from_numpy(obs_normalized).unsqueeze(0)

def get_observation_dim(env):
    obs, _ = env.reset()
    return preprocess_obs(obs).shape[1]

def safe_load_model(model_path):
    """Load checkpoint with multiple fallbacks."""
    try:
        return torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
        try:
            import pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e2:
            print(f"Method 2 (pickle) failed: {e2}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Run Sokoban with a trained PPO agent.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', help='Enable rendering for interactive play')
    parser.add_argument('--delay', type=float, default=0, help='Delay between steps in seconds (for rendering)')
    args = parser.parse_args()

    env = SokobanEnv(render_mode="human" if args.render else None)
    state_dim = get_observation_dim(env)
    action_dim = env.action_space.n

    print(f"Observation dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Load checkpoint
    checkpoint = safe_load_model(args.model)
    if checkpoint is None:
        print("Error: Could not load model file.")
        env.close()
        return

    # Determine where the model weights are stored
    model_state_dict = None
    metadata = {}

    if isinstance(checkpoint, dict):
        # Try common keys used in saved checkpoints
        for key in ['policy_state_dict', 'network_state_dict', 'state_dict']:
            if key in checkpoint:
                model_state_dict = checkpoint[key]
                metadata = checkpoint
                print(f"Found model weights under key: '{key}'")
                break
        else:
            # Maybe the top-level dict *is* the state dict?
            try:
                temp_model = ActorCritic(state_dim, action_dim)
                temp_model.load_state_dict(checkpoint)
                model_state_dict = checkpoint
                print("Checkpoint appears to be a raw state_dict.")
            except (RuntimeError, KeyError):
                pass
    elif isinstance(checkpoint, torch.nn.Module):
        # Rare: full model saved with torch.save(model)
        model_state_dict = checkpoint.state_dict()
        print("Loaded full model object.")
    else:
        print("Unknown checkpoint format.")
        env.close()
        return

    if model_state_dict is None:
        print("Error: Could not locate model state_dict in checkpoint.")
        print("Available keys:" if isinstance(checkpoint, dict) else "Checkpoint is not a dict.")
        if isinstance(checkpoint, dict):
            print(list(checkpoint.keys()))
        env.close()
        return

    # Extract dimensions from metadata if available (for compatibility)
    loaded_state_dim = metadata.get('obs_dim', metadata.get('state_dim', state_dim))
    loaded_action_dim = metadata.get('act_dim', metadata.get('action_dim', action_dim))
    hidden_dim = metadata.get('hidden_dim', 256)

    # Create model and load weights
    try:
        model = ActorCritic(loaded_state_dim, loaded_action_dim, hidden_dim)
        model.load_state_dict(model_state_dict)
    except RuntimeError as e:
        print(f"Model architecture mismatch: {e}")
        print("Ensure the saved model uses the same network structure.")
        env.close()
        return

    model.eval()
    print("✅ Model loaded successfully!")

    success_count = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        print(f"Starting episode {episode + 1}")

        while not done:
            obs_tensor = preprocess_obs(obs)
            with torch.no_grad():
                logits = model(obs_tensor)
                action = torch.argmax(logits, dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if args.render:
                env.render()
                time.sleep(args.delay)

            done = terminated or truncated

        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        success = terminated  # In Sokoban, termination usually means solved
        if success:
            success_count += 1
            status = "SUCCESS"
        else:
            status = "FAILED"
        print(f"Episode {episode + 1}: {status}, Reward = {total_reward:.2f}, Steps = {steps}")

    # Summary
    if args.episodes > 1:
        success_rate = 100 * success_count / args.episodes
        avg_len = np.mean(episode_lengths)
        avg_rew = np.mean(total_rewards)
        std_rew = np.std(total_rewards)
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Episodes: {args.episodes}")
        print(f"Success Rate: {success_rate:.2f}% ({success_count}/{args.episodes})")
        print(f"Average Episode Length: {avg_len:.2f} steps")
        print(f"Average Total Reward: {avg_rew:.2f} ± {std_rew:.2f}")
        print(f"Min/Max Reward: {min(total_rewards):.2f}/{max(total_rewards):.2f}")

    env.close()

if __name__ == "__main__":
    main()