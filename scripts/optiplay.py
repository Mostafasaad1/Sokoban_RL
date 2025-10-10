import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import warnings
from optimized_sokoban_env import OptimizedSokobanEnv

# Suppress pkg_resources deprecation warning from pygame (optional)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

class ActorCritic(nn.Module):
    """Network architecture matching the optimized training script."""
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor network - matches optimized training architecture
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 768),
            nn.Tanh(),
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        
        # Critic network - matches optimized training architecture  
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 768),
            nn.Tanh(),
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """Forward pass for policy (actor) network."""
        return self.actor(x)
    
    def get_value(self, x):
        """Get value estimate from critic network."""
        return self.critic(x)

def preprocess_obs(obs):
    """
    Preprocess observation to match optimized training format.
    REALITY: obs is (100,) with first 49 elements being the 7x7 grid, rest are padding zeros
    """
    if obs.shape != (100,):
        raise ValueError(f"Expected (100,) observation from OptimizedSokobanEnv, got {obs.shape}")
    
    # The observation is already flat and normalized by the environment
    obs_normalized = obs.astype(np.float32) / 6.0  # Normalize to [0, 1]
    return torch.from_numpy(obs_normalized).unsqueeze(0)

def extract_actual_grid(obs):
    """
    Extract the actual 7x7 game grid from the padded 100-element observation.
    Returns the meaningful 49 elements reshaped as 7x7.
    """
    if obs.shape == (100,):
        # First 49 elements are the actual 7x7 grid
        actual_data = obs[:49]
        return actual_data.reshape(7, 7)
    else:
        raise ValueError(f"Expected (100,) observation, got {obs.shape}")

def render_actual_grid(obs):
    """
    Render the ACTUAL 7x7 game grid, not the padded version.
    This shows what the game really looks like.
    """
    try:
        grid_7x7 = extract_actual_grid(obs)
        
        symbols = {
            0: '#',  # WALL
            1: ' ',  # EMPTY  
            2: '@',  # PLAYER
            3: '$',  # BOX
            4: '.',  # TARGET
            5: '*',  # BOX_ON_TARGET
            6: '+'   # PLAYER_ON_TARGET
        }
        
        print("\n🎮 ACTUAL GAME STATE (7x7 grid):")
        print("┌" + "─" * 7 + "┐")
        for row in grid_7x7:
            line = "│" + "".join(symbols.get(int(cell), '?') for cell in row) + "│"
            print(line)
        print("└" + "─" * 7 + "┘")
        
        # Show observation breakdown
        print(f"📊 Observation breakdown:")
        print(f"   • First 49 elements (7x7 grid): {obs[:49].tolist()}")
        print(f"   • Last 51 elements (padding): {obs[49:].tolist()[:10]}... (all zeros)")
        print()
        
    except Exception as e:
        print(f"❌ Error rendering grid: {e}")

def get_observation_dim(env):
    """Get the observation dimension (always 100 for OptimizedSokobanEnv)."""
    obs, _ = env.reset()
    return obs.shape[0]  # Will be 100

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
    parser = argparse.ArgumentParser(description='Run Sokoban with a trained optimized PPO agent.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', help='Enable text-based rendering')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between steps in seconds (for rendering)')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--show-padding', action='store_true', help='Show padding analysis')
    args = parser.parse_args()

    # Create environment (no render_mode parameter)
    env = OptimizedSokobanEnv()
    state_dim = get_observation_dim(env)
    action_dim = env.action_space.n

    print(f"🎯 CORRECTED Optimized Sokoban Evaluation")
    print(f"📐 Observation dimension: {state_dim} (7x7 grid + 51 padding zeros)")
    print(f"🎮 Action dimension: {action_dim}")
    print(f"🔍 REALITY: Using 7x7 game grid padded to {state_dim} elements")
    
    if args.show_padding:
        # Demonstrate the padding issue
        obs, _ = env.reset()
        print(f"\n📊 PADDING ANALYSIS:")
        print(f"   • Observation shape: {obs.shape}")
        print(f"   • Meaningful data: {np.count_nonzero(obs[:49])} non-zero values in first 49 elements")
        print(f"   • Padding data: {np.count_nonzero(obs[49:])} non-zero values in last 51 elements")
        render_actual_grid(obs)

    # Load checkpoint
    checkpoint = safe_load_model(args.model)
    if checkpoint is None:
        print("❌ Error: Could not load model file.")
        env.close()
        return

    # Determine where the model weights are stored
    model_state_dict = None
    metadata = {}

    if isinstance(checkpoint, dict):
        for key in ['policy_state_dict', 'network_state_dict', 'state_dict', 'model_state_dict']:
            if key in checkpoint:
                model_state_dict = checkpoint[key]
                metadata = checkpoint
                print(f"✅ Found model weights under key: '{key}'")
                break
        else:
            try:
                temp_model = ActorCritic(state_dim, action_dim)
                temp_model.load_state_dict(checkpoint)
                model_state_dict = checkpoint
                print("✅ Checkpoint appears to be a raw state_dict.")
            except (RuntimeError, KeyError) as e:
                print(f"❌ Failed to load as state_dict: {e}")
    elif isinstance(checkpoint, torch.nn.Module):
        model_state_dict = checkpoint.state_dict()
        print("✅ Loaded full model object.")
    else:
        print("❌ Unknown checkpoint format.")
        env.close()
        return

    if model_state_dict is None:
        print("❌ Error: Could not locate model state_dict in checkpoint.")
        if isinstance(checkpoint, dict):
            print(f"Available keys: {list(checkpoint.keys())}")
        env.close()
        return

    # Create model with correct architecture
    try:
        model = ActorCritic(state_dim, action_dim)
        model.load_state_dict(model_state_dict)
        model.eval()
        print("✅ Model loaded successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        optimal_params = sum(p.numel() for p in ActorCritic(49, action_dim).parameters())  # What it should be
        print(f"🧠 Network parameters: {total_params:,}")
        print(f"💾 Wasted parameters due to padding: {total_params - optimal_params:,} ({(total_params - optimal_params)/optimal_params*100:.1f}% overhead)")
        
    except RuntimeError as e:
        print(f"❌ Model architecture mismatch: {e}")
        env.close()
        return

    success_count = 0
    total_rewards = []
    episode_lengths = []
    solve_times = []

    print(f"\n🚀 Starting evaluation for {args.episodes} episode(s)...")
    print("=" * 70)

    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        start_time = time.time()
        
        print(f"🎮 Episode {episode + 1}/{args.episodes}")
        
        if args.render:
            print("🎬 Initial state:")
            render_actual_grid(obs)

        while not done and steps < args.max_steps:
            obs_tensor = preprocess_obs(obs)
            
            with torch.no_grad():
                logits = model(obs_tensor)
                action = torch.argmax(logits, dim=1).item()

            # Action mapping for display
            action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if args.render:
                print(f"Step {steps}: Action={action_names[action]} | Reward={reward:.3f}")
                render_actual_grid(obs)
                if args.delay > 0:
                    time.sleep(args.delay)

            done = terminated or truncated

        episode_time = time.time() - start_time
        
        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        solve_times.append(episode_time)
        
        success = terminated and not truncated
        if success:
            success_count += 1
            status = "✅ SUCCESS"
        else:
            status = "❌ FAILED" + (" (timeout)" if truncated else "")
            
        print(f"   {status} | Reward: {total_reward:.2f} | Steps: {steps} | Time: {episode_time:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)
    
    if args.episodes > 1:
        success_rate = 100 * success_count / args.episodes
        avg_len = np.mean(episode_lengths)
        avg_rew = np.mean(total_rewards)
        std_rew = np.std(total_rewards)
        avg_time = np.mean(solve_times)
        
        print(f"Episodes: {args.episodes}")
        print(f"Success Rate: {success_rate:.1f}% ({success_count}/{args.episodes})")
        print(f"Average Episode Length: {avg_len:.1f} steps")
        print(f"Average Total Reward: {avg_rew:.2f} ± {std_rew:.2f}")
        print(f"Min/Max Reward: {min(total_rewards):.2f}/{max(total_rewards):.2f}")
        print(f"Average Episode Time: {avg_time:.1f}s")
        
        if success_count > 0:
            successful_episodes = [i for i, length in enumerate(episode_lengths) if episode_lengths[i] < args.max_steps]
            if successful_episodes:
                successful_lengths = [episode_lengths[i] for i in successful_episodes]
                print(f"Average Successful Solve Length: {np.mean(successful_lengths):.1f} steps")
    else:
        print(f"Result: {'SUCCESS' if success_count > 0 else 'FAILED'}")
        print(f"Total Reward: {total_rewards[0]:.2f}")
        print(f"Episode Length: {episode_lengths[0]} steps")
        print(f"Episode Time: {solve_times[0]:.1f}s")

    print(f"\n🎯 TECHNICAL NOTES:")
    print(f"   • Environment uses 7x7 grid padded to 100 elements")
    print(f"   • Only first 49 observation elements contain game data")
    print(f"   • Network processes 51 meaningless zeros on every forward pass")
    print(f"   • Consider using true 7x7 architecture for better efficiency")

    env.close()
    print("\n🏁 Evaluation complete!")

if __name__ == "__main__":
    main()