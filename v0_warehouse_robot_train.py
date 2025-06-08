import gymnasium as gym
import numpy as np
import random
import pickle
import os
import time
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register
import v0_warehouse_robot_env  # Ensure the environment is registered

# Register the custom environment
try:
    register(
        id='warehouse-robot-v0',
        entry_point='v0_warehouse_robot_env:WarehouseRobotEnv',
    )
except:
    pass  #ignore if its already done

def train_rl(episodes=20, algorithm="q-learning", stochastic=False, gamma=0.9, epsilon_decay=0.995):
    """Trains the Warehouse Robot using Q-learning or SARSA and saves data for visualization."""
    env = gym.make('warehouse-robot-v0', stochastic=stochastic)

    q = np.zeros((env.unwrapped.grid_rows, env.unwrapped.grid_cols, 
                  env.unwrapped.grid_rows, env.unwrapped.grid_cols, 
                  3, env.action_space.n))  # 3 carrying states

    learning_rate = 0.9
    discount_factor = gamma
    epsilon = 1.0  # Initializing exploration

    total_rewards = []
    epsilon_values = []

    print(f"\n Training {algorithm.upper()} | γ={gamma} | ε-decay={epsilon_decay} for {episodes} episodes...")
    
    # printing Q table ( before traing)
    print("\n🔹 Initial Q-table (Before Training):")
    print(q)

    for episode in range(episodes):
        state, _ = env.reset()
        terminated = False
        total_reward = 0
        action = env.action_space.sample() if random.random() < epsilon else np.argmax(q[tuple(state)])

        step_count = 0
        while not terminated and step_count < 15:
            new_state, reward, terminated, _, _ = env.step(action)
            next_action = env.action_space.sample() if random.random() < epsilon else np.argmax(q[tuple(new_state)])

            q_state_action_idx = tuple(state) + (action,)
            q_new_state_idx = tuple(new_state) + (next_action,)

            if algorithm == "q-learning":
                q[q_state_action_idx] += learning_rate * (
                    reward + discount_factor * np.max(q[tuple(new_state)]) - q[q_state_action_idx]
                )
            elif algorithm == "sarsa":
                q[q_state_action_idx] += learning_rate * (
                    reward + discount_factor * q[q_new_state_idx] - q[q_state_action_idx]
                )

            state = new_state
            action = next_action
            total_reward += reward
            step_count += 1

        total_rewards.append(total_reward)
        epsilon_values.append(epsilon)
        epsilon = max(epsilon * epsilon_decay, 0.05)  # Decay epsilon

    env.close()

    # printing Q table after traing
    print("\n Trained Q-table (After Training):")
    print(q)

    # saving trained Q table
    filename = f"logs/qtable_{algorithm}_gamma{gamma}_eps{epsilon_decay}.pkl"
    os.makedirs("logs", exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(q, f)

    # Saving rewards and epsilon values for visualization in our console
    with open(f"logs/rewards_{algorithm}_gamma{gamma}_eps{epsilon_decay}.pkl", "wb") as f:
        pickle.dump(total_rewards, f)

    with open(f"logs/epsilon_{algorithm}_gamma{gamma}_eps{epsilon_decay}.pkl", "wb") as f:
        pickle.dump(epsilon_values, f)

    print(f" Training complete. Model saved as {filename}")

    return total_rewards, epsilon_values


def test_rl(algorithm="q-learning", stochastic=False, render=True):
    """Runs the trained model for 5 episodes using greedy actions and renders in Pygame."""
    
    filename = f"logs/qtable_{algorithm}_gamma0.9_eps0.995.pkl"

    if not os.path.exists(filename):
        print(f" Model '{filename}' not found. Training now...")
        train_rl(episodes=20, algorithm=algorithm, stochastic=stochastic)

    with open(filename, "rb") as f:
        q = pickle.load(f)

    env = gym.make('warehouse-robot-v0', render_mode='human' if render else None, stochastic=stochastic)

    greedy_rewards = []

    for episode in range(5):  # Running for 5 episodes
        obs, _ = env.reset()
        terminated = False
        total_reward = 0
        step_count = 0

        print(f"\n Running Episode {episode + 1} (Rendering in Pygame)")

        while not terminated and step_count < 20:
            env.render()  # Rendering an environment on each step
            time.sleep(0.3)  # Add delay for visualization
            
            action = np.argmax(q[tuple(obs)])
            obs, reward, terminated, _, _ = env.step(action)
            total_reward += reward
            step_count += 1

        greedy_rewards.append(total_reward)
        print(f"🎯 Total Reward for episode {episode + 1}: {total_reward}")

    env.close()
    return greedy_rewards


def visualize_results(results, param_type):
    """Plots total rewards and epsilon decay graphs for different hyperparameter values."""
    plt.figure(figsize=(12, 5))

    # Plot total rewards per episode for different hyperparameters
    for param, (rewards, epsilons) in results.items():
        plt.plot(rewards, marker='o', linestyle='dashed', label=f"{param_type}={param}")

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(f"Total Reward per Episode ({param_type} Comparison)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))

    # Plot epsilon decay for different hyperparameters
    for param, (rewards, epsilons) in results.items():
        plt.plot(epsilons, marker='o', linestyle='dashed', label=f"{param_type}={param}")

    plt.xlabel("Episodes")
    plt.ylabel("Epsilon Value")
    plt.title(f"Epsilon Decay ({param_type} Comparison)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    gamma_values = [0.5, 0.9, 0.99]
    epsilon_decay_values = [0.99, 0.995, 0.999]

    # Evaluating different gamma values
    gamma_results = {}
    for gamma in gamma_values:
        rewards, epsilons = train_rl(episodes=20, algorithm="q-learning", gamma=gamma, epsilon_decay=0.995)
        gamma_results[gamma] = (rewards, epsilons)

    visualize_results(gamma_results, "Gamma")

    # Evaluating different epsilon decay values
    epsilon_results = {}
    for epsilon_decay in epsilon_decay_values:
        rewards, epsilons = train_rl(episodes=20, algorithm="q-learning", gamma=0.9, epsilon_decay=epsilon_decay)
        epsilon_results[epsilon_decay] = (rewards, epsilons)

    visualize_results(epsilon_results, "Epsilon Decay")

    # Run the trained agent in Pygame
    test_rl(algorithm="q-learning", stochastic=False, render=True)

    # Suggest best hyperparameter values
    print("\n **Hyperparameter Analysis:**")
    print("- γ = 0.9 (Balanced approach, good learning speed)")
    print("- ε-decay = 0.995 (Smooth transition from exploration to exploitation)")
    print("\n **Suggested Hyperparameters: γ=0.9, ε-decay=0.995** for best performance.")