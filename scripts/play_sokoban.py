import argparse
import torch
import torch.nn as nn
import numpy as np
import time
from sokoban_env import SokobanEnv

# Use the same model architecture as in training for compatibility
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared backbone
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Policy head
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Value head (not used during inference)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared_net(x)
        action_logits = self.actor(features)
        return action_logits

def preprocess_obs(obs):
    """Match the preprocessing used during training"""
    # Flatten and normalize like in training
    obs_flat = obs.flatten()
    obs_normalized = obs_flat.astype(np.float32) / 6.0
    return torch.from_numpy(obs_normalized).unsqueeze(0)

def get_observation_dim(env):
    """Get the actual observation dimension from the environment"""
    obs, _ = env.reset()
    return preprocess_obs(obs).shape[1]  # Return feature dimension

def safe_load_model(model_path, state_dim, action_dim):
    """Safely load model with PyTorch 2.6+ compatibility"""
    try:
        # Method 1: Try with weights_only=False (requires trust in the source)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        return checkpoint
    except Exception as e:
        print(f"Method 1 failed: {e}")
        
        try:
            # Method 2: Use safe_globals context manager for PyTorch 2.6+
            import numpy.core.multiarray
            with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
                checkpoint = torch.load(model_path, map_location='cpu')
            return checkpoint
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            
            try:
                # Method 3: Try with pickle directly (fallback)
                import pickle
                with open(model_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                return checkpoint
            except Exception as e3:
                print(f"Method 3 failed: {e3}")
                return None

def main():
    parser = argparse.ArgumentParser(description='Run Sokoban with a trained PPO agent.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', help='Enable rendering for interactive play')
    parser.add_argument('--delay', type=float, default=0, help='Delay between steps in seconds (for rendering)')
    
    args = parser.parse_args()
    
    # Set up the environment first to get observation dimensions
    env = SokobanEnv(render_mode="human" if args.render else None)
    
    # Get observation and action dimensions
    state_dim = get_observation_dim(env)
    action_dim = env.action_space.n
    
    print(f"Observation dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Load the model with proper architecture
    try:
        # Use safe loading method
        checkpoint = safe_load_model(args.model, state_dim, action_dim)
        
        if checkpoint is None:
            print("All loading methods failed. Cannot load model.")
            return
            
        if 'policy_state_dict' in checkpoint:
            # This is a training checkpoint with metadata
            state_dim = checkpoint.get('state_dim', state_dim)
            action_dim = checkpoint.get('action_dim', action_dim)
            hidden_dim = checkpoint.get('hidden_dim', 256)
            
            model = ActorCritic(state_dim, action_dim, hidden_dim)
            model.load_state_dict(checkpoint['policy_state_dict'])
            print("Loaded training checkpoint with metadata")
        else:
            # This might be just the model state dict
            model = ActorCritic(state_dim, action_dim)
            model.load_state_dict(checkpoint)
            print("Loaded model state dict")
            
    except FileNotFoundError:
        print(f"Error: Model file '{args.model}' not found.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading approach...")
        # Try one more approach with the simple architecture
        try:
            model = ActorCritic(state_dim, action_dim)
            
            # Try direct loading with weights_only=False as last resort
            checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['policy_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Loaded with direct weights_only=False approach")
        except Exception as final_error:
            print(f"Final loading attempt failed: {final_error}")
            print("Cannot load the model. The file may be corrupted or in an incompatible format.")
            return
    
    model.eval()
    print("Model loaded successfully!")
    
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
            # Preprocess observation and get action
            obs_tensor = preprocess_obs(obs)
            with torch.no_grad():
                action_logits = model(obs_tensor)
                action_probs = torch.softmax(action_logits, dim=1)
                action = torch.argmax(action_logits, dim=1).item()
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Render if enabled
            if args.render:
                env.render()
                time.sleep(args.delay)
            
            # Check if the episode is done
            done = terminated or truncated
            
            # Optional: print step info
            if args.render and steps % 10 == 0:  # Print every 10 steps to avoid clutter
                print(f"Step {steps}: Action={action}, Reward={reward}, Total={total_reward}")
        
        # Track metrics
        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Check for success - Sokoban typically terminates when level is solved
        if terminated:  # In Sokoban, termination usually means success
            success_count += 1
            success_status = "SUCCESS"
        else:
            success_status = "FAILED"
        
        print(f"Episode {episode + 1}: {success_status}, Reward = {total_reward}, Steps = {steps}")
    
    # Print evaluation metrics
    if args.episodes > 1:
        success_rate = (success_count / args.episodes) * 100
        avg_length = np.mean(episode_lengths)
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Episodes: {args.episodes}")
        print(f"Success Rate: {success_rate:.2f}% ({success_count}/{args.episodes})")
        print(f"Average Episode Length: {avg_length:.2f} steps")
        print(f"Average Total Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Min/Max Reward: {min(total_rewards):.2f}/{max(total_rewards):.2f}")
    
    env.close()

if __name__ == "__main__":
    main()