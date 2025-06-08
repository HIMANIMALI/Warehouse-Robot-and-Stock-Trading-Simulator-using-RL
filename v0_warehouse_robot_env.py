import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import v0_warehouse_robot as wr

# Register the custom environment
register(
    id='warehouse-robot-v0',
    entry_point='v0_warehouse_robot_env:WarehouseRobotEnv',
)

class WarehouseRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 4}

    def __init__(self, grid_rows=6, grid_cols=6, render_mode=None, stochastic=False):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.stochastic = stochastic  # Whether the environment is stochastic

        # Initialize the WarehouseRobot problem
        self.warehouse_robot = wr.WarehouseRobot(grid_rows=grid_rows, grid_cols=grid_cols, fps=self.metadata['render_fps'], stochastic=stochastic)

        # Define the action space (LEFT, RIGHT, UP, DOWN, PICK_UP, DROP_OFF)
        self.action_space = spaces.Discrete(len(wr.RobotAction))

        # Define the observation space: [robot_row, robot_col, target_row, target_col, carrying_count]
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.grid_rows-1, self.grid_cols-1, self.grid_rows-1, self.grid_cols-1, 2]),
            shape=(5,),
            dtype=np.int64  # Change dtype to int64
        )

    def reset(self, seed=None, options=None):
        # Reset the environment
        super().reset(seed=seed)
        self.warehouse_robot.reset(seed=seed)

        # Construct the observation: [robot_row, robot_col, target_row, target_col, carrying_count]
        obs = np.concatenate((self.warehouse_robot.robot_pos, self.warehouse_robot.target_pos, [self.warehouse_robot.robot_carrying]))

        # Additional info (can be used for debugging)
        info = {}

        # Render the environment if in human mode
        if self.render_mode == 'human':
            self.render()

        return obs, info

    def step(self, action):
        # Perform the action in the WarehouseRobot environment
        reward, terminated = self.warehouse_robot.perform_action(wr.RobotAction(action))

        # Construct the new observation
        obs = np.concatenate((self.warehouse_robot.robot_pos, self.warehouse_robot.target_pos, [self.warehouse_robot.robot_carrying]))

        # Additional info (can be used for debugging)
        info = {}

        # Render the environment if in human mode
        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, False, info

    def render(self):
        # Render the WarehouseRobot environment
        self.warehouse_robot.render()

# For testing the environment
if __name__ == "__main__":
    # Deterministic environment
    print("Running deterministic environment:")
    deterministic_env = gym.make('warehouse-robot-v0', render_mode='human', stochastic=False)
    obs = deterministic_env.reset()[0]

    for _ in range(20):
        rand_action = deterministic_env.action_space.sample()
        obs, reward, terminated, _, _ = deterministic_env.step(rand_action)
        print(f"Action: {wr.RobotAction(rand_action).name}, Reward: {reward}")
        if terminated:
            print("All items delivered successfully!")
            obs = deterministic_env.reset()[0]

    # Stochastic environment
    print("\nRunning stochastic environment:")
    stochastic_env = gym.make('warehouse-robot-v0', render_mode='human', stochastic=True)
    obs = stochastic_env.reset()[0]

    for _ in range(20):
        rand_action = stochastic_env.action_space.sample()
        obs, reward, terminated, _, _ = stochastic_env.step(rand_action)
        print(f"Action: {wr.RobotAction(rand_action).name}, Reward: {reward}")
        if terminated:
            print("All items delivered successfully!")
            obs = stochastic_env.reset()[0]