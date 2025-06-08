import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from stock_trading_environment import StockTradingEnv  # Ensure this file exists

# Hyperparameters 
EPISODES = 1000  
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# Initializing environment
env = StockTradingEnv(file_path="NVDA.csv", train=True, number_of_days_to_consider=5)
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# Tracking performance
rewards_per_episode = []
epsilon_values = []

print(f" Training Q-learning agent on stock marketing trading environment for {EPISODES} episodes...")

for episode in range(EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Choose action (Exploration-Exploitation trade-off)
        if np.random.rand() < EPSILON:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit

        # Take action
        next_state, reward, done, _, _ = env.step(action)

        # Q-learning update
        q_table[state, action] += LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * np.max(q_table[next_state, :]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

    # Decay epsilon
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    rewards_per_episode.append(total_reward)
    epsilon_values.append(EPSILON)

    if (episode + 1) % 100 == 0:
        print(f" Episode {episode + 1}: Total Reward: {total_reward:.2f}, Epsilon: {EPSILON:.4f}")

# Save Q-table
with open("q_table_stock.pkl", "wb") as f:
    pickle.dump(q_table, f)

print(" Training complete. Q-table saved.")

#  Plot Total Rewards per Episode
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.grid()
plt.savefig("total_reward_stock.png")
plt.show()

# Plot Epsilon Decay
plt.figure(figsize=(10, 5))
plt.plot(epsilon_values)
plt.xlabel("Episode")
plt.ylabel("Epsilon Value")
plt.title("Epsilon Decay")
plt.grid()
plt.savefig("epsilon_decay_stock.png")
plt.show()