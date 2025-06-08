import random
from enum import Enum
import pygame
import sys
from os import path

# possible actions robot can take
class RobotAction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    PICK_UP = 4
    DROP_OFF = 5

# The Warehouse is divided into a grid.
class GridTile(Enum):
    _FLOOR = 0
    ROBOT = 1
    TARGET = 2
    ITEM = 3
    OBSTACLE = 4

    #it print object name in short robot - r
    def __str__(self):
        return self.name[:1]
    
    #The __init__ method sets up the grid, robot, obstacles, items, and Pygame rendering.
class WarehouseRobot:
    def __init__(self, grid_rows=6, grid_cols=6, fps=1, stochastic=False):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.stochastic = stochastic  # Whether the environment is stochastic
        self.reset()

        self.fps = fps
        self.last_action = ''
        self._init_pygame()
#

#_init_pygame initializes Pygame, sets up a clock, loads a font, defines 64×64 tile sizes for our images, and sets the window size.
    def _init_pygame(self):
        pygame.init()
        pygame.display.init()
        self.clock = pygame.time.Clock()
        self.action_font = pygame.font.SysFont("Calibre", 30)
        self.action_info_height = self.action_font.get_height()
        self.cell_height = 64
        self.cell_width = 64
        self.cell_size = (self.cell_width, self.cell_height)
        self.window_size = (self.cell_width * self.grid_cols, self.cell_height * self.grid_rows + self.action_info_height)
        self.window_surface = pygame.display.set_mode(self.window_size)

        # Load & resize sprites
        self.robot_img = self._load_image("robot.png")
        self.floor_img = self._load_image("floor.png")
        self.goal_img = self._load_image("goal.png")
        self.obstacle_img = self._load_image("obstacle.png")
        self.item_img = self._load_image("item.png")
        self.carrying_item_img = self._load_image("carrying_item.png")  # Image for carrying one item
        self.carrying_2_items_img = self._load_image("carrying_2_items.png")  # Image for carrying two items

    def _load_image(self, file_name):
        img = pygame.image.load(path.join(path.dirname(__file__), "sprites", file_name))
        return pygame.transform.scale(img, self.cell_size)

    def reset(self, seed=None):
        random.seed(seed)
        self.robot_pos = [0, 0]
        self.robot_carrying = 0  # 0: Not carrying, 1: Carrying one item, 2: Carrying two items
        self.target_pos = [random.randint(1, self.grid_rows-1), random.randint(1, self.grid_cols-1)]
        self.items = [[random.randint(1, self.grid_rows-1), random.randint(1, self.grid_cols-1)] for _ in range(2)]
        self.obstacles = [[2, 2], [3, 3], [4, 4]]

    def perform_action(self, robot_action: RobotAction):
        self.last_action = robot_action
        new_pos = self.robot_pos.copy()
        reward = -1  # Default reward for each step
        terminated = False

        # Handle stochastic behavior
        if self.stochastic and robot_action in [RobotAction.LEFT, RobotAction.RIGHT, RobotAction.UP, RobotAction.DOWN]:
            if random.random() < 0.1:  # 10% chance to fail the action
                new_pos = self.robot_pos  # Stay in the same position
                reward -= 20  # Penalty for failing the action
            else:
                # Perform the action as intended
                if robot_action == RobotAction.LEFT:
                    new_pos[1] = max(0, new_pos[1] - 1)
                elif robot_action == RobotAction.RIGHT:
                    new_pos[1] = min(self.grid_cols - 1, new_pos[1] + 1)
                elif robot_action == RobotAction.UP:
                    new_pos[0] = max(0, new_pos[0] - 1)
                elif robot_action == RobotAction.DOWN:
                    new_pos[0] = min(self.grid_rows - 1, new_pos[0] + 1)
        else:
            # Deterministic behavior
            if robot_action == RobotAction.LEFT:
                new_pos[1] = max(0, new_pos[1] - 1)
            elif robot_action == RobotAction.RIGHT:
                new_pos[1] = min(self.grid_cols - 1, new_pos[1] + 1)
            elif robot_action == RobotAction.UP:
                new_pos[0] = max(0, new_pos[0] - 1)
            elif robot_action == RobotAction.DOWN:
                new_pos[0] = min(self.grid_rows - 1, new_pos[0] + 1)

        # Handle pick-up and drop-off actions
        if robot_action == RobotAction.PICK_UP:
            if self.robot_carrying < 2 and new_pos in self.items:
                self.robot_carrying += 1  # Increment carrying count
                self.items.remove(new_pos)  # Remove the item from the grid
                if self.robot_carrying == 1:
                    reward += 25  # Reward for picking up the first item
                elif self.robot_carrying == 2:
                    reward += 200  # Reward for picking up both items
        elif robot_action == RobotAction.DROP_OFF:
            if self.robot_carrying > 0 and new_pos == self.target_pos:
                if self.robot_carrying == 1:
                    reward += 100  # Reward for delivering one item
                elif self.robot_carrying == 2:
                    reward += 500  # Reward for delivering both items
                self.robot_carrying = 0  # Reset carrying count
                if not self.items:  # If all items are delivered
                    terminated = True  # Episode ends

        # Check for collisions with obstacles
        if new_pos in self.obstacles:
            reward -= 20  # Penalty for hitting an obstacle
        else:
            self.robot_pos = new_pos

        return reward, terminated

    def render(self):
        self._process_events()
        self.window_surface.fill((255, 255, 255))

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                pos = (c * self.cell_width, r * self.cell_height)
                self.window_surface.blit(self.floor_img, pos)

                if [r, c] in self.obstacles:
                    self.window_surface.blit(self.obstacle_img, pos)
                if [r, c] in self.items:
                    self.window_surface.blit(self.item_img, pos)
                if [r, c] == self.target_pos:
                    self.window_surface.blit(self.goal_img, pos)
                if [r, c] == self.robot_pos:
                    # Display the appropriate robot image based on carrying state
                    if self.robot_carrying == 1:
                        self.window_surface.blit(self.carrying_item_img, pos)
                    elif self.robot_carrying == 2:
                        self.window_surface.blit(self.carrying_2_items_img, pos)
                    else:
                        self.window_surface.blit(self.robot_img, pos)

        text_img = self.action_font.render(f'Action: {self.last_action}', True, (0, 0, 0), (255, 255, 255))
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos)

        pygame.display.update()
        self.clock.tick(self.fps)

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    # Create a deterministic environment
    deterministic_robot = WarehouseRobot(stochastic=False)
    print("Running deterministic environment:")
    deterministic_robot.render()

    for _ in range(50):
        rand_action = random.choice(list(RobotAction))
        print(f"Action: {rand_action}")
        reward, terminated = deterministic_robot.perform_action(rand_action)
        print(f"Reward: {reward}")
        deterministic_robot.render()
        if terminated:
            print("All items delivered successfully!")
            break

    # Create a stochastic environment
    stochastic_robot = WarehouseRobot(stochastic=True)
    print("\nRunning stochastic environment:")
    stochastic_robot.render()

    for _ in range(50):
        rand_action = random.choice(list(RobotAction))
        print(f"Action: {rand_action}")
        reward, terminated = stochastic_robot.perform_action(rand_action)
        print(f"Reward: {reward}")
        stochastic_robot.render()
        if terminated:
            print("All items delivered successfully!")
            break