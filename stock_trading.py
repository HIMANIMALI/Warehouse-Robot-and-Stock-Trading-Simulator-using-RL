import pickle
import numpy as np
from stock_trading_environment import StockTradingEnv

# Load trained Q-table
with open("q_table_stock.pkl", "rb") as f:
    q_table = pickle.load(f)

# Initialize environment in evaluation mode
env = StockTradingEnv(file_path="NVDA.csv", train=False, number_of_days_to_consider=5)
state, _ = env.reset()
terminated = False

print("Evaluating trained agent using greedy actions...")

while not terminated:
    action = np.argmax(q_table[state, :])  # Always choose best action
    next_state, _, terminated, _, _ = env.step(action)
    state = next_state

# Render final portfolio value over time
env.render()