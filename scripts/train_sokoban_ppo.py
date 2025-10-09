# scripts/train_sokoban_ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import time
import os
from tqdm import tqdm
import sys

# Add parent directory to path to import sokoban_env
sys.path.append('..')
from sokoban_env import SokobanEnv

class PPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, use_one_hot=True):
        super().__init__()
        self.use_one_hot = use_one_hot
        self.num_tile_types = 7  # WALL, EMPTY, PLAYER, BOX, TARGET, BOX_ON_TARGET, PLAYER_ON_TARGET
        
        if use_one_hot:
            input_dim = obs_dim * self.num_tile_types
        else:
            input_dim = obs_dim
        
        print(f"Network input dimension: {input_dim}")
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0.0)
    
    def preprocess_obs(self, obs):
        """Convert observation to suitable format for network"""
        if self.use_one_hot:
            # Convert to one-hot encoding
            obs_tensor = torch.tensor(obs, dtype=torch.long)
            one_hot = torch.nn.functional.one_hot(obs_tensor, num_classes=self.num_tile_types)
            return one_hot.float().view(-1)  # Flatten
        else:
            # Normalize to [0, 1]
            return torch.tensor(obs, dtype=torch.float32) / 6.0
    
    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = self.preprocess_obs(obs)
        
        features = self.shared_net(obs)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value
    
    def get_action(self, obs):
        with torch.no_grad():
            logits, value = self.forward(obs)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item(), value.item()
    
    def evaluate_actions(self, obs, actions):
        logits, value = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value

class PPOBuffer:
    def __init__(self, obs_dim, size, gamma=0.99, gae_lambda=0.95):
        # Store raw observations, we'll preprocess during training
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.int32)
        self.act_buf = np.zeros(size, dtype=np.int32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
    
    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
    
    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.gae_lambda)
        
        # Rewards-to-go
        self.ret_buf[path_slice] = self._discount_cumsum(rews[:-1], self.gamma)
        
        self.path_start_idx = self.ptr
    
    def _discount_cumsum(self, x, discount):
        return np.array([np.sum(x[i:] * (discount ** np.arange(len(x) - i))) 
                        for i in range(len(x))])
    
    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        
        # Advantage normalization
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        return (self.obs_buf, 
                self.act_buf,
                self.ret_buf,
                self.adv_buf,
                self.logp_buf)

class PPOAgent:
    def __init__(self, env, hidden_dim=256, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, train_epochs=4, batch_size=64, use_one_hot=True):
        self.env = env
        
        # Get actual observation dimensions by resetting the environment
        obs, _ = self.env.reset()
        self.obs_dim = len(obs)
        self.act_dim = env.action_space.n
        
        print(f"Detected observation dimension: {self.obs_dim}")
        print(f"Action dimension: {self.act_dim}")
        
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.use_one_hot = use_one_hot
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.network = PPONetwork(
            self.obs_dim, self.act_dim, hidden_dim, use_one_hot
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Training stats
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        self.training_steps = 0
    
    def train(self, total_timesteps=5_000_000, rollout_length=128, 
              save_freq=100_000, checkpoint_path="sokoban_ppo.pth"):
        
        buffer = PPOBuffer(self.obs_dim, rollout_length, self.gamma, self.gae_lambda)
        
        # Load checkpoint if exists
        start_step = 0
        if os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
            start_step = self.training_steps
            print(f"Resumed training from step {start_step}")
        
        obs, _ = self.env.reset()
        ep_return, ep_length = 0, 0
        success_count = 0
        
        progress_bar = tqdm(total=total_timesteps, initial=start_step, desc="Training")
        
        for step in range(start_step, total_timesteps):
            # Collect rollout
            for _ in range(rollout_length):
                action, logp, val = self.network.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                buffer.store(obs, action, reward, val, logp)
                
                obs = next_obs
                ep_return += reward
                ep_length += 1
                
                if done:
                    # Check if episode was successful
                    success = (reward > 0.5)  # Sokoban gives +1.0 reward for solving
                    if success:
                        success_count += 1
                    
                    self.episode_returns.append(ep_return)
                    self.episode_lengths.append(ep_length)
                    self.success_rate.append(success)
                    
                    # Logging
                    if len(self.episode_returns) >= 10 and len(self.episode_returns) % 10 == 0:
                        avg_return = np.mean(self.episode_returns)
                        avg_length = np.mean(self.episode_lengths)
                        current_success_rate = np.mean(self.success_rate)
                        tqdm.write(f"Step {step}: Avg Return: {avg_return:.2f}, "
                                 f"Avg Length: {avg_length:.1f}, "
                                 f"Success Rate: {current_success_rate:.2f}")
                    
                    obs, _ = self.env.reset()
                    ep_return, ep_length = 0, 0
            
            # After collecting rollout
            with torch.no_grad():
                _, _, last_val = self.network.get_action(obs)
            buffer.finish_path(last_val)
            
            # Update policy
            self.update(buffer)
            
            progress_bar.update(rollout_length)
            self.training_steps += rollout_length
            
            # Save checkpoint
            if self.training_steps % save_freq == 0:
                self.save_checkpoint(checkpoint_path)
                print(f"\nCheckpoint saved at step {self.training_steps}")
        
        progress_bar.close()
        self.save_checkpoint(checkpoint_path)
        print("Training completed!")
    
    def update(self, buffer):
        obs, acts, rets, advs, old_logps = buffer.get()
        
        # Convert to tensors and move to device
        obs_tensor = torch.tensor(obs, dtype=torch.int32)
        acts_tensor = torch.tensor(acts, dtype=torch.long).to(self.device)
        rets_tensor = torch.tensor(rets, dtype=torch.float32).to(self.device)
        advs_tensor = torch.tensor(advs, dtype=torch.float32).to(self.device)
        old_logps_tensor = torch.tensor(old_logps, dtype=torch.float32).to(self.device)
        
        # Preprocess all observations at once for efficiency
        if self.network.use_one_hot:
            obs_processed = []
            for ob in obs_tensor:
                obs_processed.append(self.network.preprocess_obs(ob.numpy()))
            obs_processed = torch.stack(obs_processed).to(self.device)
        else:
            obs_processed = obs_tensor.float().to(self.device) / 6.0
        
        # PPO update epochs
        for _ in range(self.train_epochs):
            # Shuffle for minibatch updates
            indices = torch.randperm(len(obs_processed))
            
            for start in range(0, len(obs_processed), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                batch_obs = obs_processed[idx]
                batch_acts = acts_tensor[idx]
                batch_rets = rets_tensor[idx]
                batch_advs = advs_tensor[idx]
                batch_old_logps = old_logps_tensor[idx]
                
                # Get current policy and value
                logps, entropy, values = self.network.evaluate_actions(batch_obs, batch_acts)
                
                # Policy loss
                ratio = torch.exp(logps - batch_old_logps)
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advs
                policy_loss = -torch.min(ratio * batch_advs, clip_adv).mean()
                
                # Value loss
                value_loss = 0.5 * ((values.squeeze() - batch_rets) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + value_loss + 0.01 * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
    
    def save_checkpoint(self, path):
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'episode_returns': list(self.episode_returns),
            'episode_lengths': list(self.episode_lengths),
            'success_rate': list(self.success_rate),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.episode_returns = deque(checkpoint['episode_returns'], maxlen=100)
        self.episode_lengths = deque(checkpoint['episode_lengths'], maxlen=100)
        self.success_rate = deque(checkpoint['success_rate'], maxlen=100)

def debug_environment():
    """Debug function to check environment properties"""
    env = SokobanEnv(render_mode=None)
    obs, _ = env.reset()
    
    print("=== Environment Debug Info ===")
    print(f"Observation shape: {obs.shape}")
    print(f"Observation type: {type(obs)}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Observation sample: {obs[:10]}...")  # First 10 elements
    print(f"Unique values in obs: {np.unique(obs)}")
    
    # Test a few steps
    for i in range(3):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        print(f"Step {i}: action={action}, reward={reward}, done={terminated or truncated}")
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    
    env.close()

def main():
    # First, debug the environment to understand the observation space
    debug_environment()
    
    # Create environment
    env = SokobanEnv(render_mode=None)  # No rendering during training
    
    # Initialize agent - start with raw observations (no one-hot) for simplicity
    agent = PPOAgent(
        env,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        train_epochs=4,
        batch_size=64,
        use_one_hot=False  # Start with raw observations to avoid shape issues
    )
    
    # Train the agent
    print("Starting PPO training on Sokoban...")
    
    agent.train(
        total_timesteps=5_000_000,
        rollout_length=128,
        save_freq=100_000,
        checkpoint_path="sokoban_ppo.pth"
    )

if __name__ == "__main__":
    main()