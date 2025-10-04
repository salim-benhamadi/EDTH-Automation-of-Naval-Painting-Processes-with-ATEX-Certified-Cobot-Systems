"""
Step 3: RL agent that learns to paint the object
"""
import pyvista as pv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
import os

class PaintingEnvironment(gym.Env):
    """RL environment for painting a 3D object"""
    
    def __init__(self, mesh_path):
        super().__init__()
        
        self.mesh = pv.read(mesh_path)
        self.n_points = self.mesh.n_points
        
        # State: painted status of each vertex (0 = unpainted, 1 = painted)
        self.painted = np.zeros(self.n_points)
        
        # Action space: which vertex to paint next
        self.action_space = spaces.Discrete(self.n_points)
        
        # Observation space: current painting state
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_points,), dtype=np.float32
        )
        
        self.steps = 0
        self.max_steps = MAX_STEPS
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.painted = np.zeros(self.n_points)
        self.steps = 0
        return self.painted.copy(), {}
    
    def step(self, action):
        # Paint the selected vertex and neighbors
        self.painted[action] = 1.0
        
        # Paint nearby vertices (brush effect)
        distances = np.linalg.norm(
            self.mesh.points - self.mesh.points[action], axis=1
        )
        nearby = distances < 0.02  # 2cm radius
        self.painted[nearby] = 1.0
        
        self.steps += 1
        
        # Calculate reward
        coverage = self.painted.mean()
        efficiency = coverage / (self.steps / self.max_steps)
        reward = coverage * 10 + efficiency * 5
        
        # Check if done
        done = coverage > 0.95 or self.steps >= self.max_steps
        truncated = self.steps >= self.max_steps
        
        return self.painted.copy(), reward, done, truncated, {"coverage": coverage}
    
    def render(self):
        """Visualize current painting state"""
        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh, scalars=self.painted, cmap='RdYlGn', 
                        clim=[0, 1], show_edges=False)
        plotter.add_text(f"Coverage: {self.painted.mean():.1%}", position='upper_left')
        plotter.show()

class PolicyNetwork(nn.Module):
    """Simple neural network for policy"""
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

def train_agent(mesh_path, episodes=EPISODES):
    """Train RL agent to paint object"""
    print(f"\n{'='*50}")
    print("Training RL Painting Agent")
    print(f"{'='*50}")
    
    env = PaintingEnvironment(mesh_path)
    
    # Simple policy gradient agent
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_rewards = []
        log_probs = []
        
        done = False
        while not done:
            # Select action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
            
            # Sample action from probabilities
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Take step
            state, reward, done, truncated, info = env.step(action.item())
            
            episode_rewards.append(reward)
            log_probs.append(log_prob)
            
            if done or truncated:
                break
        
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Update policy
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()
        
        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards) / len(episode_rewards)
            coverage = info.get('coverage', 0)
            print(f"Episode {episode+1}/{episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Coverage: {coverage:.1%}")
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = f"{MODELS_DIR}/painting_agent.pth"
    torch.save(policy.state_dict(), model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    return policy

def test_agent(mesh_path, model_path=None):
    """Test trained agent and visualize result"""
    print("\nTesting agent...")
    
    env = PaintingEnvironment(mesh_path)
    
    if model_path is None:
        model_path = f"{MODELS_DIR}/painting_agent.pth"
    
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = policy(state_tensor)
        
        action = torch.argmax(probs).item()
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done or truncated:
            break
    
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Coverage: {info['coverage']:.1%}")
    
    # Save painted mesh
    os.makedirs(PAINTED_DIR, exist_ok=True)
    painted_mesh = env.mesh.copy()
    painted_mesh['painted'] = env.painted
    output_path = f"{PAINTED_DIR}/painted_object.vtk"
    painted_mesh.save(output_path)
    print(f"✓ Painted mesh saved to {output_path}")
    
    # Visualize
    env.render()

if __name__ == "__main__":
    import sys
    
    object_name = sys.argv[1] if len(sys.argv) > 1 else TARGET_OBJECT
    mesh_path = f"{FILTERED_DIR}/{object_name}_clean.obj"
    
    # Train
    train_agent(mesh_path)
    
    # Test
    test_agent(mesh_path)