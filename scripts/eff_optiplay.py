"""
UNIVERSAL SOKOBAN VISUALIZER
============================

This visualizer can handle ALL Sokoban model types:
- Original models (your base format)
- Optimized models (100-element padded)
- Efficient models (49-element direct)

Automatically detects model type and loads with correct architecture!
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try importing all possible environments
try:
    from sokoban_env import SokobanEnv
    ORIGINAL_ENV_AVAILABLE = True
except ImportError:
    ORIGINAL_ENV_AVAILABLE = False
    print("⚠️ Original SokobanEnv not available")

try:
    from optimized_sokoban_env import OptimizedSokobanEnv
    OPTIMIZED_ENV_AVAILABLE = True
except ImportError:
    OPTIMIZED_ENV_AVAILABLE = False
    print("⚠️ OptimizedSokobanEnv not available")

try:
    from efficient_sokoban_env import EfficientSokobanEnv
    EFFICIENT_ENV_AVAILABLE = True
except ImportError:
    EFFICIENT_ENV_AVAILABLE = False
    print("⚠️ EfficientSokobanEnv not available")

# =============================================================================
# NETWORK ARCHITECTURES
# =============================================================================

class ActorCritic(nn.Module):
    """Original ActorCritic architecture"""
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

class AdvancedPPONetwork(nn.Module):
    """Advanced PPO network from optimized system"""
    def __init__(self, obs_shape, num_actions, hidden_size=768):
        super().__init__()
        input_size = obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape)
        
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        shared_out = self.shared(x.float())
        logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return logits, value.squeeze(-1)

class EfficientPPONetwork(nn.Module):
    """Efficient PPO network (49 inputs)"""
    def __init__(self, obs_shape, num_actions, hidden_size=512):
        super().__init__()
        input_size = obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape)
        
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        shared_out = self.shared(x.float())
        logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return logits, value.squeeze(-1)

# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def preprocess_obs_original(obs):
    """Original preprocessing (normalize by 6.0)"""
    obs_flat = obs.flatten()
    obs_normalized = obs_flat.astype(np.float32) / 6.0
    return torch.from_numpy(obs_normalized).unsqueeze(0)

def preprocess_obs_optimized(obs):
    """Optimized preprocessing (already flat, convert to float)"""
    obs_normalized = obs.astype(np.float32)
    return torch.from_numpy(obs_normalized).unsqueeze(0)

def preprocess_obs_efficient(obs):
    """Efficient preprocessing (already flat, convert to float)"""
    obs_normalized = obs.astype(np.float32)
    return torch.from_numpy(obs_normalized).unsqueeze(0)

# =============================================================================
# ENVIRONMENT MANAGEMENT
# =============================================================================

class EnvironmentManager:
    """Manages different environment types"""
    
    def __init__(self, model_type, render_mode=None):
        self.model_type = model_type
        self.env = None
        self.preprocess_fn = None
        
        if model_type == "original":
            if ORIGINAL_ENV_AVAILABLE:
                self.env = SokobanEnv(render_mode=render_mode)
                self.preprocess_fn = preprocess_obs_original
            else:
                raise ValueError("Original SokobanEnv not available")
                
        elif model_type == "optimized":
            if OPTIMIZED_ENV_AVAILABLE:
                self.env = OptimizedSokobanEnv()
                self.preprocess_fn = preprocess_obs_optimized
            else:
                raise ValueError("OptimizedSokobanEnv not available")
                
        elif model_type == "efficient":
            if EFFICIENT_ENV_AVAILABLE:
                self.env = EfficientSokobanEnv()
                self.preprocess_fn = preprocess_obs_efficient
            else:
                raise ValueError("EfficientSokobanEnv not available")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        if hasattr(self.env, 'render'):
            self.env.render()
        elif hasattr(self.env, 'render_grid'):
            print(self.env.render_grid())
        else:
            print("Rendering not available for this environment")
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def get_action_space_n(self):
        return self.env.action_space.n
    
    def preprocess_obs(self, obs):
        return self.preprocess_fn(obs)

# =============================================================================
# MODEL DETECTION AND LOADING
# =============================================================================

def detect_model_type(model_path):
    """
    Smart model type detection based on checkpoint contents
    """
    print(f"🔍 Analyzing model: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Check for efficient model metadata
        if 'obs_shape' in checkpoint and 'num_actions' in checkpoint:
            obs_shape = checkpoint['obs_shape']
            if obs_shape[0] == 49:
                print(f"✅ Detected: EFFICIENT model (49-element observations)")
                return "efficient", checkpoint
            elif obs_shape[0] == 100:
                print(f"✅ Detected: OPTIMIZED model (100-element observations)")
                return "optimized", checkpoint
        
        # Check state dict structure
        if 'network_state_dict' in checkpoint:
            state_dict = checkpoint['network_state_dict']
        elif 'policy_state_dict' in checkpoint:
            state_dict = checkpoint['policy_state_dict']
        else:
            state_dict = checkpoint
        
        # Analyze first layer to determine input size
        if 'shared.0.weight' in state_dict:
            input_size = state_dict['shared.0.weight'].shape[1]
            if input_size == 49:
                print(f"✅ Detected: EFFICIENT model (49 inputs)")
                return "efficient", checkpoint
            elif input_size == 100:
                print(f"✅ Detected: OPTIMIZED model (100 inputs)")
                return "optimized", checkpoint
            else:
                print(f"✅ Detected: ORIGINAL model ({input_size} inputs)")
                return "original", checkpoint
                
        elif 'shared_net.0.weight' in state_dict:
            input_size = state_dict['shared_net.0.weight'].shape[1]
            print(f"✅ Detected: ORIGINAL model ({input_size} inputs)")
            return "original", checkpoint
            
        else:
            print(f"⚠️ Unknown model structure, defaulting to ORIGINAL")
            return "original", checkpoint
            
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        return "original", None

def load_model(model_path, model_type, checkpoint, action_dim):
    """
    Load model with appropriate architecture based on detected type
    """
    print(f"🧠 Loading {model_type.upper()} model...")
    
    try:
        if model_type == "original":
            # Get state dict
            if 'policy_state_dict' in checkpoint:
                state_dict = checkpoint['policy_state_dict']
                state_dim = checkpoint.get('state_dim', 49)  # Default fallback
                hidden_dim = checkpoint.get('hidden_dim', 256)
            else:
                state_dict = checkpoint
                # Try to infer from state dict
                if 'shared_net.0.weight' in state_dict:
                    state_dim = state_dict['shared_net.0.weight'].shape[1]
                else:
                    state_dim = 49  # Default
                hidden_dim = 256
            
            model = ActorCritic(state_dim, action_dim, hidden_dim)
            model.load_state_dict(state_dict)
            obs_shape = (state_dim,)
            
        elif model_type == "optimized":
            # Advanced PPO network
            if 'obs_shape' in checkpoint:
                obs_shape = checkpoint['obs_shape']
            else:
                # Infer from state dict
                state_dict = checkpoint.get('network_state_dict', checkpoint)
                input_size = state_dict['shared.0.weight'].shape[1]
                obs_shape = (input_size,)
            
            model = AdvancedPPONetwork(obs_shape, action_dim, hidden_size=768)
            model.load_state_dict(checkpoint.get('network_state_dict', checkpoint))
            
        elif model_type == "efficient":
            # Efficient PPO network
            if 'obs_shape' in checkpoint:
                obs_shape = checkpoint['obs_shape']
            else:
                obs_shape = (49,)  # Default for efficient
            
            model = EfficientPPONetwork(obs_shape, action_dim, hidden_size=512)
            model.load_state_dict(checkpoint.get('network_state_dict', checkpoint))
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.eval()
        print(f"✅ Model loaded successfully!")
        print(f"📐 Input shape: {obs_shape}")
        print(f"🎮 Actions: {action_dim}")
        
        return model, obs_shape
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# =============================================================================
# MAIN VISUALIZATION FUNCTION
# =============================================================================

def run_visualization(model, env_manager, args):
    """
    Run the visualization with performance tracking
    """
    success_count = 0
    total_rewards = []
    episode_lengths = []
    
    print(f"\n🎮 Starting visualization with {args.episodes} episodes...")
    print("="*60)
    
    for episode in range(args.episodes):
        obs, _ = env_manager.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n📺 Episode {episode + 1}")
        
        # Track timing
        step_times = []
        inference_times = []
        render_times = []
        
        if args.render:
            print("Initial State:")
            env_manager.render()
        
        while not done:
            step_start = time.time()
            
            # Preprocess observation
            obs_tensor = env_manager.preprocess_obs(obs)
            
            # Model inference
            inference_start = time.time()
            with torch.no_grad():
                output = model(obs_tensor)
                
                # Handle different model output formats
                if isinstance(output, tuple):
                    action_logits, value = output
                else:
                    action_logits = output
                    value = None
                
                action = torch.argmax(action_logits, dim=1).item()
            inference_end = time.time()
            
            # Step environment
            obs, reward, terminated, truncated, info = env_manager.step(action)
            total_reward += reward
            steps += 1
            
            # Render
            render_start = time.time()
            if args.render:
                if not args.no_render_every_step or steps % 10 == 0 or terminated or truncated:
                    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                    print(f"\nStep {steps}: Action {action_names[action]}")
                    if value is not None:
                        print(f"Value: {value.item():.3f}")
                    env_manager.render()
                    print(f"Reward: {reward:.3f} | Total: {total_reward:.3f}")
                    
                    if args.delay > 0:
                        time.sleep(args.delay)
            render_end = time.time()
            
            step_end = time.time()
            
            # Track timing
            step_times.append(step_end - step_start)
            inference_times.append(inference_end - inference_start)
            render_times.append(render_end - render_start)
            
            done = terminated or truncated
            
            # Progress update
            if not args.render and steps % 50 == 0:
                print(f"  Step {steps}: Total reward = {total_reward:.2f}")
        
        # Episode summary
        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if terminated:
            success_count += 1
            success_status = "✅ SUCCESS"
        else:
            success_status = "❌ FAILED"
        
        # Performance stats
        avg_step_time = np.mean(step_times) if step_times else 0
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        avg_render_time = np.mean(render_times) if render_times else 0
        
        print(f"Episode {episode + 1}: {success_status}")
        print(f"  Reward: {total_reward:.3f} | Steps: {steps}")
        print(f"  Timing: Step={avg_step_time:.4f}s, Inference={avg_inference_time:.4f}s, Render={avg_render_time:.4f}s")
    
    # Final summary
    if args.episodes > 1:
        success_rate = (success_count / args.episodes) * 100
        avg_length = np.mean(episode_lengths)
        avg_reward = np.mean(total_rewards)
        
        print("\n" + "="*60)
        print("🏆 EVALUATION SUMMARY")
        print("="*60)
        print(f"Episodes: {args.episodes}")
        print(f"Success Rate: {success_rate:.2f}% ({success_count}/{args.episodes})")
        print(f"Average Episode Length: {avg_length:.2f} steps")
        print(f"Average Total Reward: {avg_reward:.2f}")
        print(f"Best Reward: {max(total_rewards):.2f}")
        print("="*60)
    
    return {
        'success_rate': success_count / args.episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_steps': np.mean(episode_lengths),
        'total_episodes': args.episodes
    }

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Universal Sokoban Model Visualizer')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', help='Enable rendering for interactive play')
    parser.add_argument('--delay', type=float, default=0, help='Delay between steps in seconds (for rendering)')
    parser.add_argument('--fast', action='store_true', help='Enable fast mode (minimal rendering overhead)')
    parser.add_argument('--no-render-every-step', action='store_true', help='Skip rendering every step for faster execution')
    parser.add_argument('--force-type', type=str, choices=['original', 'optimized', 'efficient'], 
                        help='Force specific model type (skip auto-detection)')
    
    args = parser.parse_args()
    
    print("🚀 UNIVERSAL SOKOBAN VISUALIZER")
    print("="*50)
    print(f"📁 Model: {args.model}")
    print(f"🎮 Episodes: {args.episodes}")
    print(f"👁️ Render: {args.render}")
    print(f"⏱️ Delay: {args.delay}s")
    
    # Detect model type
    if args.force_type:
        model_type = args.force_type
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
        print(f"🔧 Forced model type: {model_type.upper()}")
    else:
        model_type, checkpoint = detect_model_type(args.model)
    
    if checkpoint is None:
        print("❌ Failed to load model checkpoint")
        return
    
    # Create environment manager
    try:
        render_mode = "human" if (args.render and model_type == "original") else None
        env_manager = EnvironmentManager(model_type, render_mode)
        action_dim = env_manager.get_action_space_n()
        print(f"🌍 Environment: {model_type.upper()}")
        print(f"🎯 Action space: {action_dim}")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return
    
    # Load model
    model, obs_shape = load_model(args.model, model_type, checkpoint, action_dim)
    if model is None:
        print("❌ Failed to load model")
        return
    
    # Run visualization
    print("\n" + "="*50)
    try:
        results = run_visualization(model, env_manager, args)
        print("\n✅ Visualization complete!")
    except KeyboardInterrupt:
        print("\n🛑 Visualization interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
    finally:
        env_manager.close()

if __name__ == "__main__":
    main()