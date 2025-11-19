from replay_buffer import ReplayBuffer
import numpy as np

def main():
    buf = ReplayBuffer(capacity=5)

    # Fake transitions: states are just numbers 0,1,2,3,...
    for i in range(7):
        state = np.array([i, i+1, i+2, i+3], dtype=np.float32)  # fake 4-dim state
        action = i % 2
        reward = float(i)
        next_state = state + 1
        done = (i % 3 == 0)
        buf.push(state, action, reward, next_state, done)
        print(f"Pushed transition {i}, buffer size now {len(buf)}")

    # Sample 3 transitions
    states, actions, rewards, next_states, dones = buf.sample(3)
    print("Sampled states shape:", states.shape)
    print("Sampled actions:", actions)
    print("Sampled rewards:", rewards)
    print("Sampled dones:", dones)

if __name__ == "__main__":
    main()
