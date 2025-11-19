import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from networks import QNetwork
from replay_buffer import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_Agent:
    def __init__(self,
                obs_dim,
                action_dim,
                gamma = 0.99,
                lr = 1e-3,
                epsilon_start = 1.0,
                epsilon_end = 0.05,
                epsilon_decay = 500,
                buffer_capacity = 100_000,
                batch_size = 64,
                target_update_freq = 100):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0
        self.target_update_freq = target_update_freq
        
        #Initialize networks
        self.q_net = QNetwork(obs_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(obs_dim, action_dim).to(DEVICE)
        
        #Initialize target network
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr = lr)
        
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
    def _epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.total_steps/self.epsilon_decay)
    
    def select_action(self, state):
        eps = self._epsilon()
        self.total_steps += 1
        
        if random.random() < eps:
            return random.randrange(self.action_dim)
        
        state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        action = int(q_values.argmax(dim = 1).item())
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states_t = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=DEVICE)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        
        q_values = self.q_net(states_t).gather(1, actions_t)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim = 1, keepdim = True)[0]

            targets = rewards_t + (1.0 - dones_t) * self.gamma * next_q_values
            
        loss = nn.functional.mse_loss(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()
    
    
def train_dqn(
    env_id = "CartPole-v1",
    num_episodes = 500,
    max_steps_per_episode=500):
    
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n 
    
    agent = DQN_Agent(obs_dim, action_dim)
    
    returns = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_return = 0.0
        
        for t in range(max_steps_per_episode):
            action = agent.select_action(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done  = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            
            loss = agent.train_step()
            
            state = next_state
            
            episode_return += reward

            if done:
                break
        returns.append(episode_return)

        if (episode + 1) % 10 == 0:
            avg_return = np.mean(returns[-10:])
            
            print(f"Episode {episode+1}/{num_episodes} | Return={episode_return:.1f} | Avg(10)={avg_return:.1f}")
            
    env.close()
    
    return agent, returns

def evaluate(agent, env_id="CartPole-v1", n_episodes=5):
    env = gym.make(env_id, render_mode=None)
    returns = []

    for ep in range(n_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Greedy action: no epsilon, no randomness
            state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.q_net(state_t)
            action = int(q_values.argmax(dim=1).item())

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        returns.append(total_reward)
        print(f"[EVAL] Episode {ep+1}, return={total_reward}")

    env.close()
    print(f"[EVAL] Average return over {n_episodes} episodes: {np.mean(returns):.1f}")
    return returns


def watch_agent(agent, env_id="CartPole-v1", n_episodes=3):
    env = gym.make(env_id, render_mode="human")

    for ep in range(n_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Pure greedy action
            state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.q_net(state_t)
            action = int(q_values.argmax(dim=1).item())

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        print(f"[WATCH] Episode {ep+1}, return={total_reward}")

    env.close()



if __name__ == "__main__":
    agent, returns = train_dqn()
    evaluate(agent)
    watch_agent(agent)