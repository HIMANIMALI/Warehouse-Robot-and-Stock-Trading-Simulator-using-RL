import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class StockTradingEnv(gym.Env):
    """A simple Stock marketing Trading environment for Q-learning."""
    
    def __init__(self, file_path, train=True, number_of_days_to_consider=5):
        super(StockTradingEnv, self).__init__()
        self.data = pd.read_csv(file_path)
        self.train = train
        self.number_of_days_to_consider = number_of_days_to_consider
        self.current_step = number_of_days_to_consider
        self.initial_balance = 100000  # Starting balance we have
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.balance
        self.terminated = False
        self.portfolio_values = []  # use for Storing portfolio values over time

        # Actions: 0 = Buy, 1 = Sell, 2 = Hold
        self.action_space = spaces.Discrete(3)

        # Observations: (price increase, stock held)
        self.observation_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        self.current_step = self.number_of_days_to_consider
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.balance
        self.terminated = False
        self.portfolio_values = []  # Reseting portfolio tracking

        return self._get_observation(), {}

    def step(self, action):
        previous_price = self.data.iloc[self.current_step - 1]["Close"]
        current_price = self.data.iloc[self.current_step]["Close"]

        reward = 0
        if action == 0:  # Buy
            if self.balance >= current_price:
                self.shares_held = self.balance // current_price
                self.balance -= self.shares_held * current_price
        elif action == 1:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0
        elif action == 2:  # Hold
            pass

        self.total_value = self.balance + (self.shares_held * current_price)
        reward = self.total_value - self.initial_balance

        # **Track portfolio value over time**
        self.portfolio_values.append(self.total_value)

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.terminated = True

        return self._get_observation(), reward, self.terminated, False, {}

    def _get_observation(self):
        price_increase = self.data.iloc[self.current_step]["Close"] > self.data.iloc[self.current_step - self.number_of_days_to_consider]["Close"]
        stock_held = self.shares_held > 0

        if price_increase and not stock_held:
            return 0
        elif price_increase and stock_held:
            return 1
        elif not price_increase and stock_held:
            return 2
        else:
            return 3

    def render(self):
        """Ploting Portfolio Value Over Time"""
        if self.portfolio_values:
            plt.figure(figsize=(10, 5))
            plt.plot(self.portfolio_values, marker='o', linestyle='dashed', label="Portfolio Value")
            plt.xlabel("Steps")
            plt.ylabel("Portfolio Value ($)")
            plt.title("Agent's Portfolio Value Over Time")
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print(" No portfolio data to visualize.")