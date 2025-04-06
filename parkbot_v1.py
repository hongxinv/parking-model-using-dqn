from enum import Enum
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import time

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class ParkingLot(gym.Env): #by passing env to the ParkingLot class, we are inheriting methods from the gym environment cass
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode = None, size = 3):
        # The size of the parking lot
        self.size = size # 3x3 square for now
        self.window_size = 512 # size of PyGame Display window
        
        # Observations are dictionaries with the agent's and the target's location
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        
        # Car Starting location
        self._agent_location = np.array([0, 1], dtype = int)
        self._target_location = np.array([2, 0], dtype = int)
        
        # actions car can take, for now there is no concept of direction car is facing
        self.action_space = spaces.Discrete(4)
        
        # Note: Origin is top-left in this gridworld
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.DOWN.value: np.array([0, 1]),
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.UP.value: np.array([0, -1]),
        }
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        # initialising to None saves resources until they need to be used
        
        self.window = None
        self.clock = None
        
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    # returns manhattan distance from the goal, may be useful for future
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }    
    
    def step(self, action):
        # Mpa the actions (0,1,2,3) to the direction the car moves
        direction = self._action_to_direction[action]
        # 'np.clip' makes sure agent stays in the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size -1
        )
        # An episode is done iff the agent has reached the target
        # terminated is a boolean of whether agent has reached target
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        # +1 for correct parking, -1 for time spent, -3 for collisions/being in the wrong parking spot
        reward = 1 if terminated else 0 if self._agent_location[0] == 1 else -3
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
            
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels
            
        pix_square_size = self.window_size / self.size
        
        agent_image = pygame.image.load("car.webp")
        agent_image = pygame.transform.scale(
            agent_image, (int(pix_square_size), int(pix_square_size))
        )
        
        agent_pos = (self._agent_location * pix_square_size). astype(int)
        
        canvas.blit(agent_image, agent_pos)
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()



        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # # Now we draw the agent
        # pygame.draw.circle(
        #     canvas,
        #     (0, 0, 255),
        #     (self._agent_location + 0.5) * pix_square_size,
        #     pix_square_size / 3,
        # )
        
        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
            
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def reset(self, seed = None, options = None):
        
        # ensure rng is seeded properly
        super().reset(seed = seed)
        
        # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype = int)
        
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        #     if not np.array_equal(self._target_location, self._agent_location) and self._target_location[1] != 1:
        #         break

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

## Simulation ##         
env = ParkingLot(render_mode="human", size=3)  # Enable human rendering
obs, info = env.reset()

cumu_reward = 0
for _ in range(10):  # Run 10 steps
    print("Start Simulation")
    action = env.action_space.sample()  # Random action (replace with your policy later)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    cumu_reward += reward
    
    print(f"Action: {action}, Reward: {reward}, Cumulative Reward: {cumu_reward} Terminated: {terminated}")
    
    if terminated or truncated:
        break

    time.sleep(0.0001)  # optional: slow down so you can see it better

env.close()