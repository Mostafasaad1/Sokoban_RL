import argparse
import torch
import torch.nn as nn
import numpy as np
import time
from sokoban_env import SokobanEnv

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared_net(x)
        action_logits = self.actor(features)
        return action_logits

def preprocess_obs(obs):
    obs_flat = obs.flatten()
    obs_normalized = obs_flat.astype(np.float32) / 6.0
    return torch.from_numpy(obs_normalized).unsqueeze(0)

def get_observation_dim(env):
    obs, _ = env.reset()
    return preprocess_obs(obs).shape[1]

def safe_load_model(model_path, state_dim, action_dim):
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        return checkpoint
    except Exception as e:
        print(f"Method 1 failed: {e}")
        try:
            import numpy.core.multiarray
            with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
                checkpoint = torch.load(model_path, map_location='cpu')
            return checkpoint
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            try:
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
    parser.add_argument('--fast', action='store_true', help='Enable fast mode (minimal rendering overhead)')
    parser.add_argument('--no-render-every-step', action='store_true', help='Skip rendering every step for faster execution')
    
    args = parser.parse_args()
    
    # Set up the environment
    render_mode = "human" if args.render else None
    if args.fast and args.render:
        print("Fast mode enabled - using minimal rendering")
    
    env = SokobanEnv(render_mode=render_mode)
    
    # Get observation and action dimensions
    state_dim = get_observation_dim(env)
    action_dim = env.action_space.n
    
    print(f"Observation dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Load the model
    try:
        checkpoint = safe_load_model(args.model, state_dim, action_dim)
        
        if checkpoint is None:
            print("All loading methods failed. Cannot load model.")
            return
            
        if 'policy_state_dict' in checkpoint:
            state_dim = checkpoint.get('state_dim', state_dim)
            action_dim = checkpoint.get('action_dim', action_dim)
            hidden_dim = checkpoint.get('hidden_dim', 256)
            model = ActorCritic(state_dim, action_dim, hidden_dim)
            model.load_state_dict(checkpoint['policy_state_dict'])
            print("Loaded training checkpoint with metadata")
        else:
            model = ActorCritic(state_dim, action_dim)
            model.load_state_dict(checkpoint)
            print("Loaded model state dict")
            
    except FileNotFoundError:
        print(f"Error: Model file '{args.model}' not found.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            model = ActorCritic(state_dim, action_dim)
            checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['policy_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Loaded with direct weights_only=False approach")
        except Exception as final_error:
            print(f"Final loading attempt failed: {final_error}")
            return
    
    model.eval()
    print("Model loaded successfully!")
    
    success_count = 0
    total_rewards = []
    episode_lengths = []
    
    # Pre-compile any expensive operations if possible
    softmax = torch.nn.Softmax(dim=1)
    
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"Starting episode {episode + 1}")
        
        # Track timing for performance monitoring
        step_times = []
        inference_times = []
        render_times = []
        
        while not done:
            step_start = time.time()
            
            # Preprocess observation and get action
            obs_tensor = preprocess_obs(obs)
            
            # Model inference
            inference_start = time.time()
            with torch.no_grad():
                action_logits = model(obs_tensor)
                action = torch.argmax(action_logits, dim=1).item()
            inference_end = time.time()
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Conditional rendering for performance
            render_start = time.time()
            if args.render:
                if not args.no_render_every_step or steps % 10 == 0 or terminated or truncated:
                    env.render()
                    if args.delay > 0:
                        time.sleep(args.delay)
            render_end = time.time()
            
            step_end = time.time()
            
            # Track timing
            step_times.append(step_end - step_start)
            inference_times.append(inference_end - inference_start)
            render_times.append(render_end - render_start)
            
            # Check if the episode is done
            done = terminated or truncated
            
            # Minimal printing during execution
            if args.render and steps % 1 == 0:
                print(f"Step {steps}: Total reward = {total_reward}")
        
        # Track metrics
        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if terminated:
            success_count += 1
            success_status = "SUCCESS"
        else:
            success_status = "FAILED"
        
        # Print performance statistics
        avg_step_time = np.mean(step_times)
        avg_inference_time = np.mean(inference_times)
        avg_render_time = np.mean(render_times)
        
        print(f"Episode {episode + 1}: {success_status}, Reward = {total_reward}, Steps = {steps}")
        print(f"Performance: Step={avg_step_time:.4f}s, Inference={avg_inference_time:.4f}s, Render={avg_render_time:.4f}s")
    
    # Print evaluation metrics
    if args.episodes > 1:
        success_rate = (success_count / args.episodes) * 100
        avg_length = np.mean(episode_lengths)
        avg_reward = np.mean(total_rewards)
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Episodes: {args.episodes}")
        print(f"Success Rate: {success_rate:.2f}% ({success_count}/{args.episodes})")
        print(f"Average Episode Length: {avg_length:.2f} steps")
        print(f"Average Total Reward: {avg_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    main()