# improved_train_sokoban_ppo.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from improved_sokoban_env import ImprovedSokobanEnv
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


class ImprovedActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(ImprovedActorCritic, self).__init__()
        
        # Larger shared backbone with dropout for regularization
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Policy head with proper initialization
        self.actor = nn.Linear(hidden_dim // 2, action_dim)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)  # Small initialization for exploration
        
        # Value head  
        self.critic = nn.Linear(hidden_dim // 2, 1)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        
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


class ImprovedPPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, ppo_epochs=4, hidden_dim=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.policy = ImprovedActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Use different learning rates for actor and critic
        actor_params = list(self.policy.actor.parameters())
        critic_params = list(self.policy.critic.parameters()) + list(self.policy.shared_net.parameters())
        
        self.actor_optimizer = optim.Adam(actor_params, lr=lr)
        self.critic_optimizer = optim.Adam(critic_params, lr=lr * 2)  # Critic learns faster
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        
        self.buffer = RolloutBuffer()
        
        # Adaptive learning
        self.learning_rate = lr
        self.initial_lr = lr
        
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
    
    def update(self, total_steps=0):
        if len(self.buffer.states) == 0:
            return 0, 0, 0
            
        states, actions, old_log_probs, rewards, values, dones = self.buffer.get_tensors(self.device)
        
        # Compute advantages and returns (all on GPU)
        advantages, returns = self.compute_advantages(rewards, values, dones)
        
        # Adaptive learning rate
        lr_decay = max(0.1, 1.0 - total_steps / 1_000_000)  # Decay to 10% over 1M steps
        current_lr = self.initial_lr * lr_decay
        
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = current_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = current_lr * 2
        
        policy_loss_avg = 0
        value_loss_avg = 0
        entropy_loss_avg = 0
        
        # PPO epochs with mini-batching
        batch_size = max(64, len(states) // 4)  # Adaptive batch size
        indices = torch.randperm(len(states))
        
        for epoch in range(self.ppo_epochs):
            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get new action probabilities and values
                _, new_log_probs, entropy, new_values = self.policy.get_action(batch_states, batch_actions)
                new_values = new_values.squeeze()
                
                # Policy ratio
                ratio = (new_log_probs - batch_old_log_probs).exp()
                
                # Policy loss with clipping
                policy_loss1 = ratio * batch_advantages
                policy_loss2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                
                # Value loss with clipping
                value_loss = 0.5 * (batch_returns - new_values).pow(2).mean()
                
                # Entropy bonus - adaptive
                entropy_coef = max(0.01, 0.1 - total_steps / 2_000_000)  # Decay from 0.1 to 0.01
                entropy_loss = -entropy.mean()
                
                # Separate updates for actor and critic
                actor_loss = policy_loss + entropy_coef * entropy_loss
                critic_loss = value_loss
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.critic.parameters()) + list(self.policy.shared_net.parameters()), 0.5
                )
                self.critic_optimizer.step()
                
                policy_loss_avg += policy_loss.item()
                value_loss_avg += value_loss.item()
                entropy_loss_avg += entropy_loss.item()
        
        # Clear buffer
        self.buffer.clear()
        
        num_updates = self.ppo_epochs * max(1, len(states) // batch_size)
        return (policy_loss_avg / num_updates, 
                value_loss_avg / num_updates, 
                entropy_loss_avg / num_updates)
    
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
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
        self.policy = ImprovedActorCritic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        if 'actor_optimizer_state_dict' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        if 'critic_optimizer_state_dict' in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


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
    if num_envs > 1:
        max_dim = 0
        for single_env in env.envs:
            for _ in range(3):
                obs, _ = single_env.reset()
                dim = obs.flatten().shape[0]
                max_dim = max(max_dim, dim)
        return max_dim
    else:
        obs, _ = env.reset()
        return obs.flatten().shape[0]


class EpisodeTracker:
    """Enhanced episode tracking for improved monitoring"""
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.episode_rewards = [0.0] * num_envs
        self.episode_lengths = [0] * num_envs
        self.completed_rewards = []
        self.completed_lengths = []
        self.completed_successes = []
        self.boxes_on_targets_final = []
        
    def step(self, rewards, dones, infos):
        """Update tracking with step results"""
        for i in range(self.num_envs):
            self.episode_rewards[i] += rewards[i]
            self.episode_lengths[i] += 1
            
            if dones[i]:
                self.completed_rewards.append(self.episode_rewards[i])
                self.completed_lengths.append(self.episode_lengths[i])
                
                # Check success and final boxes on targets
                final_info = infos.get('final_info', [None] * self.num_envs)[i]
                if final_info and 'boxes_on_targets' in final_info:
                    boxes_on_targets = final_info['boxes_on_targets']
                    is_success = boxes_on_targets >= 2  # Both boxes on targets
                    self.boxes_on_targets_final.append(boxes_on_targets)
                else:
                    is_success = False
                    self.boxes_on_targets_final.append(0)
                
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
            'avg_boxes_on_targets': np.mean(self.boxes_on_targets_final[-n:]),
            'total_episodes': len(self.completed_rewards),
            'recent_rewards': self.completed_rewards[-n:],
            'recent_successes': self.completed_successes[-n:]
        }


def train_improved_sokoban(args):
    # Environment setup - vectorized
    if args.num_envs > 1:
        # Determine max observation size first
        temp_env = ImprovedSokobanEnv()
        max_obs_size = 0
        for _ in range(10):
            obs, _ = temp_env.reset()
            max_obs_size = max(max_obs_size, obs.flatten().shape[0])
        temp_env.close()
        
        env = PaddedSyncVectorEnv(
            [lambda: ImprovedSokobanEnv() for _ in range(args.num_envs)],
            max_obs_size=max_obs_size
        )
        print(f"Created {args.num_envs} improved vectorized environments (max obs size: {max_obs_size})")
        state_dim = max_obs_size
    else:
        env = ImprovedSokobanEnv()
        print("Using single improved environment (num_envs=1)")
        state_dim = get_observation_dim(env, args.num_envs)
    
    action_dim = env.action_space.n if args.num_envs == 1 else env.single_action_space.n
    
    print(f"Observation dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Initialize improved PPO agent
    agent = ImprovedPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs,
        hidden_dim=args.hidden_dim
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
    pbar = tqdm(total=args.total_timesteps, initial=start_step, desc="Improved Training")
    
    # Training loop
    obs, info = env.reset()
    state = preprocess_observation(obs).to(agent.device)
    
    print("🚀 Starting improved training with better reward shaping...")
    
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
            
            # Print enhanced statistics
            stats = tracker.get_stats(last_n=20)
            if stats and stats['total_episodes'] % 20 == 0 and len(tracker.completed_rewards) > 0:
                last_logged = getattr(train_improved_sokoban, 'last_logged_episode', 0)
                if stats['total_episodes'] > last_logged:
                    print(f"\n{'='*80}")
                    print(f"📊 Episode {stats['total_episodes']} | Steps: {total_steps:,}")
                    print(f"🏆 Success Rate (last 20): {stats['success_rate']:.1%}")
                    print(f"💰 Avg Reward (last 20): {stats['avg_reward']:.3f}")
                    print(f"⏱️  Avg Length (last 20): {stats['avg_length']:.1f}")
                    print(f"📦 Avg Boxes on Targets: {stats['avg_boxes_on_targets']:.1f}/2")
                    print(f"🎯 Recent Success Pattern: {['✅' if s > 0 else '❌' for s in stats['recent_successes'][-10:]]}")
                    print(f"{'='*80}")
                    train_improved_sokoban.last_logged_episode = stats['total_episodes']
        
        # Update policy
        if len(agent.buffer.states) > 0:
            policy_loss, value_loss, entropy_loss = agent.update(total_steps)
            
            # Log losses occasionally
            if total_steps % 10000 < args.num_envs * args.rollout_steps and total_steps > 0:
                print(f"\n{'~'*60}")
                print(f"📈 Step {total_steps:,}")
                print(f"🎭 Policy Loss: {policy_loss:.6f}")
                print(f"💎 Value Loss: {value_loss:.6f}")
                print(f"🎲 Entropy Loss: {entropy_loss:.6f}")
                print(f"🧠 Learning Rate: {agent.actor_optimizer.param_groups[0]['lr']:.6f}")
                print(f"{'~'*60}")
        
        # Save model periodically
        if total_steps % args.save_interval < args.num_envs * args.rollout_steps and total_steps > 0:
            agent.save(args.model_path)
            print(f"\n💾 Model saved at step {total_steps:,}")
    
    # Final save
    agent.save(args.model_path)
    pbar.close()
    env.close()
    
    # Print final statistics
    stats = tracker.get_stats(last_n=100)
    if stats:
        print(f"\n{'='*80}")
        print("🎉 IMPROVED TRAINING COMPLETE")
        print(f"📊 Total Episodes: {stats['total_episodes']:,}")
        print(f"🏆 Final Success Rate (last 100): {stats['success_rate']:.1%}")
        print(f"💰 Final Avg Reward (last 100): {stats['avg_reward']:.3f}")
        print(f"📦 Final Avg Boxes on Targets: {stats['avg_boxes_on_targets']:.1f}/2")
        print(f"{'='*80}")


def evaluate_improved_sokoban(args):
    """Evaluate the improved trained policy with rendering"""
    env = ImprovedSokobanEnv(render_mode="human")
    
    # Load model to get dimensions first
    if not os.path.exists(args.model_path):
        print("No trained model found!")
        return
        
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dim = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']
    hidden_dim = checkpoint.get('hidden_dim', 512)
    
    agent = ImprovedPPO(state_dim, action_dim, hidden_dim=hidden_dim)
    agent.load(args.model_path)
    print(f"Loaded improved model from {args.model_path}")
    
    for episode in range(5):  # Test 5 episodes
        state, info = env.reset()
        state = preprocess_observation(state)
        
        episode_reward = 0
        steps = 0
        max_steps = 200
        
        print(f"\n🎮 Episode {episode + 1}")
        
        while steps < max_steps:
            env.render()
            time.sleep(0.2)
            
            with torch.no_grad():
                state_tensor = state.unsqueeze(0).to(agent.device).float()
                action_tensor, _, _, _ = agent.policy.get_action(state_tensor)
                action = action_tensor.item()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            state = preprocess_observation(next_state)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                boxes_on_targets = info.get('boxes_on_targets', 0)
                print(f"📊 Episode {episode + 1} Results:")
                print(f"   💰 Reward: {episode_reward:.3f}")
                print(f"   ⏱️  Steps: {steps}")
                print(f"   📦 Boxes on targets: {boxes_on_targets}/2")
                if terminated and boxes_on_targets >= 2:
                    print("   🎉 SUCCESS: All boxes on targets!")
                elif truncated:
                    print("   ⏰ Episode truncated")
                else:
                    print("   ❌ Episode ended without success")
                break
        
        if steps >= max_steps:
            print(f"   ⏰ Reached maximum steps ({max_steps}) without completing")
        
        time.sleep(1)  # Pause between episodes
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Improved PPO on Sokoban")
    parser.add_argument("--render", action="store_true", help="Render the trained policy")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")  # Lower LR
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ppo-epochs", type=int, default=8, help="PPO epochs per update")  # More epochs
    parser.add_argument("--rollout-steps", type=int, default=256, help="Steps per env per rollout")  # Longer rollouts
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="Total training steps")
    parser.add_argument("--save-interval", type=int, default=50000, help="Save model every N steps")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension for neural network")
    parser.add_argument("--model-path", type=str, default="improved_sokoban_ppo.pth", help="Model save path")
    
    args = parser.parse_args()
    
    if args.render:
        evaluate_improved_sokoban(args)
    else:
        train_improved_sokoban(args)