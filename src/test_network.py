if __name__ == "__main__":
    import gymnasium as gym
    from networks import QNetwork
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n   

    net = QNetwork(obs_dim, action_dim)
    print(net)

    # Fake batch of 3 states
    import torch
    x = torch.randn(3, obs_dim)
    q_values = net(x)
    print("Input shape:", x.shape)
    print("Output shape:", q_values.shape) 
    env.close()
