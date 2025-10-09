# scripts/train_sokoban_ppo.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.vector.utils import concatenate, create_empty_array
from sokoban_env import SokobanEnv
from tqdm import tqdm
import time


class PaddedSyncVectorEnv(SyncVectorEnv):
    """SyncVectorEnv that handles variable-sized observations with padding"""
    
    def __init__(self, env_fns, max_obs_size=None):
        super().__init__(env_fns)
        self.max_obs_size = max_obs_size
        
    def reset(self, seed=None, options=None):
        """Reset all environments and pad observations"""
        if seed is not None:
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            seeds = [None] * self.num_envs
            
        observations = []
        infos = {}
        
        for i, (env, single_seed) in enumerate(zip(self.envs, seeds)):
            kwargs = {}
            if single_seed is not None:
                kwargs['seed'] = single_seed
            if options is not None:
                kwargs['options'] = options
                
            obs, info = env.reset(**kwargs)
            observations.append(obs)
            infos = self._add_info(infos, info, i)
        
        # Pad observations to max size
        self._observations = self._pad_observations(observations)
        
        return self._observations, infos
    
    def step(self, actions):
        """Step all environments and pad observations"""
        observations = []
        rewards = np.zeros(self.num_envs)
        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)
        infos = {}
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards[i] = reward
            terminations[i] = terminated
            truncations[i] = truncated
            infos = self._add_info(infos, info, i)
            
            # Auto-reset if done
            if terminated or truncated:
                final_obs = obs
                final_info = info
                obs, info = env.reset()
                observations[i] = obs
                
                # Add final observation to info
                info['final_observation'] = final_obs
                info['final_info'] = final_info
                infos = self._add_info(infos, info, i)
        
        # Pad observations to max size
        self._observations = self._pad_observations(observations)
        
        return self._observations, rewards, terminations, truncations, infos
    
    def _pad_observations(self, observations):
        """Pad all observations to max_obs_size"""
        if self.max_obs_size is None:
            # Determine max size from current batch
            self.max_obs_size = max(obs.flatten().shape[0] for obs in observations)
        
        padded = np.zeros((len(observations), self.max_obs_size), dtype=np.float32)
        for i, obs in enumerate(observations):
            flat_obs = obs.flatten()
            padded[i, :len(flat_obs)] = flat_obs
        
        return padded
    
    def _add_info(self, infos, info, i):
        """Add info from environment i to the infos dict"""
        for key, value in info.items():
            if key not in infos:
                infos[key] = [None] * self.num_envs
            infos[key][i] = value
        return infos


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
        
        # Value head  
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        features = self.shared_net(x)
        return self.actor(features), self.critic(features)
    
    def get_action(self, x, action=None):
        logits, value = self.forward(x)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def add(self, state, action, log_prob, reward, value, done):
        """Add batched transitions (num_envs,)"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
    def get_tensors(self, device):
        """Return tensors directly on device, flattened across envs and time"""
        # Stack: [T, num_envs, ...] -> Flatten: [T * num_envs, ...]
        states = torch.stack(self.states).view(-1, self.states[0].shape[-1])
        actions = torch.stack(self.actions).view(-1)
        log_probs = torch.stack(self.log_probs).view(-1)
        rewards = torch.stack(self.rewards).view(-1)
        values = torch.stack(self.values).view(-1)
        dones = torch.stack(self.dones).view(-1)
        
        return (states.to(device), actions.to(device), log_probs.to(device),
                rewards.to(device), values.to(device), dones.to(device))


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, ppo_epochs=4, hidden_dim=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        
        self.buffer = RolloutBuffer()
        
    def compute_advantages(self, rewards, values, dones):
        """GPU-only advantage computation using GAE"""
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        next_value = 0
        
        # Compute advantages using GAE (reversed loop)
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        if len(self.buffer.states) == 0:
            return 0, 0, 0
            
        states, actions, old_log_probs, rewards, values, dones = self.buffer.get_tensors(self.device)
        
        # Compute advantages and returns (all on GPU)
        advantages, returns = self.compute_advantages(rewards, values, dones)
        
        policy_loss_avg = 0
        value_loss_avg = 0
        entropy_loss_avg = 0
        
        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Get new action probabilities and values
            _, new_log_probs, entropy, new_values = self.policy.get_action(states, actions)
            new_values = new_values.squeeze()
            
            # Policy ratio
            ratio = (new_log_probs - old_log_probs).exp()
            
            # Policy loss with clipping
            policy_loss1 = ratio * advantages
            policy_loss2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
            
            # Value loss
            value_loss = 0.5 * (returns - new_values).pow(2).mean()
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            policy_loss_avg += policy_loss.item()
            value_loss_avg += value_loss.item()
            entropy_loss_avg += entropy_loss.item()
        
        # Clear buffer
        self.buffer.clear()
        
        return (policy_loss_avg / self.ppo_epochs, 
                value_loss_avg / self.ppo_epochs, 
                entropy_loss_avg / self.ppo_epochs)
    
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        
        # Recreate policy with correct dimensions
        self.policy = ActorCritic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def preprocess_observation(obs, max_size=None):
    """Convert observation(s) to float32 tensor, handling both single and batched"""
    if isinstance(obs, np.ndarray) and obs.ndim == 2:
        # Already batched and padded from PaddedSyncVectorEnv
        obs_normalized = obs.astype(np.float32) / 6.0
    elif isinstance(obs, (list, tuple)):
        # Batched observations - need to pad
        flattened = []
        for o in obs:
            flat = o.flatten().astype(np.float32) / 6.0
            if max_size is not None and len(flat) < max_size:
                padded = np.zeros(max_size, dtype=np.float32)
                padded[:len(flat)] = flat
                flattened.append(padded)
            else:
                flattened.append(flat[:max_size] if max_size else flat)
        obs_normalized = np.array(flattened, dtype=np.float32)
    else:
        # Single observation
        obs_flat = obs.flatten().astype(np.float32) / 6.0
        if max_size is not None and len(obs_flat) < max_size:
            obs_normalized = np.zeros(max_size, dtype=np.float32)
            obs_normalized[:len(obs_flat)] = obs_flat
        else:
            obs_normalized = obs_flat[:max_size] if max_size else obs_flat
    
    return torch.from_numpy(obs_normalized)


def get_observation_dim(env, num_envs=1):
    """Get the actual observation dimension from the environment"""
    # For vectorized envs, we need to determine the maximum observation size
    if num_envs > 1:
        # Access individual environments directly to avoid stacking issues
        max_dim = 0
        for single_env in env.envs:
            for _ in range(3):  # Sample a few times
                obs, _ = single_env.reset()
                dim = obs.flatten().shape[0]
                max_dim = max(max_dim, dim)
        return max_dim
    else:
        obs, _ = env.reset()
        return obs.flatten().shape[0]


class EpisodeTracker:
    """Track episode statistics across vectorized environments"""
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.episode_rewards = [0.0] * num_envs
        self.episode_lengths = [0] * num_envs
        self.completed_rewards = []
        self.completed_lengths = []
        self.completed_successes = []
    
    def step(self, rewards, dones, infos):
        """Update tracking with step results"""
        for i in range(self.num_envs):
            self.episode_rewards[i] += rewards[i]
            self.episode_lengths[i] += 1
            
            if dones[i]:
                self.completed_rewards.append(self.episode_rewards[i])
                self.completed_lengths.append(self.episode_lengths[i])
                # Check if episode was successful (terminated, not truncated)
                is_success = infos.get('final_info', [None] * self.num_envs)[i] is not None
                self.completed_successes.append(1.0 if is_success else 0.0)
                
                # Reset for this env
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
    
    def get_stats(self, last_n=10):
        """Get statistics from last N completed episodes"""
        if len(self.completed_rewards) == 0:
            return None
        
        n = min(last_n, len(self.completed_rewards))
        return {
            'avg_reward': np.mean(self.completed_rewards[-n:]),
            'avg_length': np.mean(self.completed_lengths[-n:]),
            'success_rate': np.mean(self.completed_successes[-n:]),
            'total_episodes': len(self.completed_rewards)
        }


def train_sokoban(args):
    # Environment setup - vectorized
    if args.num_envs > 1:
        # Determine max observation size first
        temp_env = SokobanEnv()
        max_obs_size = 0
        for _ in range(10):
            obs, _ = temp_env.reset()
            max_obs_size = max(max_obs_size, obs.flatten().shape[0])
        temp_env.close()
        
        env = PaddedSyncVectorEnv(
            [lambda: SokobanEnv() for _ in range(args.num_envs)],
            max_obs_size=max_obs_size
        )
        print(f"Created {args.num_envs} vectorized environments (max obs size: {max_obs_size})")
        state_dim = max_obs_size
    else:
        env = SokobanEnv()
        print("Using single environment (num_envs=1)")
        state_dim = get_observation_dim(env, args.num_envs)
    
    action_dim = env.action_space.n if args.num_envs == 1 else env.single_action_space.n
    
    print(f"Observation dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Initialize PPO agent
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        agent.load(args.model_path)
        start_step = 0
    
    # Episode tracking
    tracker = EpisodeTracker(args.num_envs)
    total_steps = 0
    
    # Progress bar
    pbar = tqdm(total=args.total_timesteps, initial=start_step, desc="Training")
    
    # Training loop
    obs, info = env.reset()
    state = preprocess_observation(obs).to(agent.device)
    
    while total_steps < args.total_timesteps:
        # Collect rollout
        for _ in range(args.rollout_steps):
            with torch.no_grad():
                # Get actions for all envs
                action_tensor, log_prob, _, value = agent.policy.get_action(state)
                
                if args.num_envs > 1:
                    actions = action_tensor.cpu().numpy()
                else:
                    actions = action_tensor.item()
            
            # Step all environments
            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            # Handle single vs vectorized env outputs
            if args.num_envs == 1:
                rewards = np.array([rewards])
                terminateds = np.array([terminateds])
                truncateds = np.array([truncateds])
                dones = np.array([terminateds or truncateds])
            else:
                dones = np.logical_or(terminateds, truncateds)
            
            # Convert to tensors for buffer
            rewards_tensor = torch.from_numpy(np.array(rewards)).float()
            dones_tensor = torch.from_numpy(np.array(dones)).float()
            
            # Store transition
            agent.buffer.add(
                state,
                action_tensor,
                log_prob,
                rewards_tensor,
                value.squeeze(),
                dones_tensor
            )
            
            # Update episode tracking
            tracker.step(rewards, dones, infos)
            
            # Update state
            state = preprocess_observation(next_obs).to(agent.device)
            
            # Update step count
            steps_increment = args.num_envs
            total_steps += steps_increment
            pbar.update(steps_increment)
            
            # Print statistics every 10 completed episodes
            stats = tracker.get_stats(last_n=10)
            if stats and stats['total_episodes'] % 10 == 0 and len(tracker.completed_rewards) > 0:
                last_logged = getattr(train_sokoban, 'last_logged_episode', 0)
                if stats['total_episodes'] > last_logged:
                    print(f"\n{'='*60}")
                    print(f"Episode {stats['total_episodes']}")
                    print(f"Avg Reward (last 10): {stats['avg_reward']:.2f}")
                    print(f"Avg Length (last 10): {stats['avg_length']:.2f}")
                    print(f"Success Rate (last 10): {stats['success_rate']:.2%}")
                    print(f"Total Steps: {total_steps}")
                    print(f"{'='*60}")
                    train_sokoban.last_logged_episode = stats['total_episodes']
        
        # Update policy
        if len(agent.buffer.states) > 0:
            policy_loss, value_loss, entropy_loss = agent.update()
            
            # Log losses occasionally
            if total_steps % 10000 < args.num_envs * args.rollout_steps and total_steps > 0:
                print(f"\n{'~'*60}")
                print(f"Step {total_steps}")
                print(f"Policy Loss: {policy_loss:.4f}")
                print(f"Value Loss: {value_loss:.4f}")
                print(f"Entropy Loss: {entropy_loss:.4f}")
                print(f"{'~'*60}")
        
        # Save model periodically
        if total_steps % args.save_interval < args.num_envs * args.rollout_steps and total_steps > 0:
            agent.save(args.model_path)
            print(f"\n💾 Model saved at step {total_steps}")
    
    # Final save
    agent.save(args.model_path)
    pbar.close()
    env.close()
    
    # Print final statistics
    stats = tracker.get_stats(last_n=100)
    if stats:
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"Total Episodes: {stats['total_episodes']}")
        print(f"Final Avg Reward (last 100): {stats['avg_reward']:.2f}")
        print(f"Final Success Rate (last 100): {stats['success_rate']:.2%}")
        print(f"{'='*60}")


def evaluate_sokoban(args):
    """Evaluate the trained policy with rendering (single env only)"""
    env = SokobanEnv()
    
    # Load model to get dimensions first
    if not os.path.exists(args.model_path):
        print("No trained model found!")
        return
        
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dim = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']
    hidden_dim = checkpoint['hidden_dim']
    
    agent = PPO(state_dim, action_dim, hidden_dim=hidden_dim)
    agent.load(args.model_path)
    print(f"Loaded model from {args.model_path}")
    
    state, info = env.reset()
    state = preprocess_observation(state)
    
    episode_reward = 0
    steps = 0
    max_steps = 1000  # Prevent infinite loops
    
    while steps < max_steps:
        env.render()
        time.sleep(0.1)  # Slow down for visualization
        
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(agent.device).float()
            action_tensor, _, _, _ = agent.policy.get_action(state_tensor)
            action = action_tensor.item()
        
        next_state, reward, terminated, truncated, info = env.step(action)
        state = preprocess_observation(next_state)
        
        episode_reward += reward
        steps += 1
        
        if terminated or truncated:
            print(f"Episode finished! Reward: {episode_reward}, Steps: {steps}")
            if terminated:
                print("SUCCESS: All boxes on targets!")
            else:
                print("Episode truncated")
            break
    
    if steps >= max_steps:
        print(f"Reached maximum steps ({max_steps}) without completing the level")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on Sokoban")
    parser.add_argument("--render", action="store_true", help="Render the trained policy")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--rollout-steps", type=int, default=128, help="Steps per env per rollout")
    parser.add_argument("--total-timesteps", type=int, default=5000000, help="Total training steps")
    parser.add_argument("--save-interval", type=int, default=100000, help="Save model every N steps")
    parser.add_argument("--model-path", type=str, default="sokoban_ppo_gymnasium.pth", help="Model save path")
    
    args = parser.parse_args()
    
    if args.render:
        evaluate_sokoban(args)
    else:
        train_sokoban(args)