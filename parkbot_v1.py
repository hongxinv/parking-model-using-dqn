from enum import Enum
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import time

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random

# Action Space
class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

# Facing Directions
class FacingDirection(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

# The direction the agent is facing in

# DQN
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args)) #saves transition

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Simulation
class ParkingLot(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=3):
        self.size = size
        self.window_size = 512
        self.blocks = 3 # (3x3square) x 3times

        self.full_width = self.size * self.blocks
        self.tile_size = 64
        self.window_width = self.tile_size * self.full_width
        self.window_height = self.tile_size * self.size

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(0, self.full_width - 1, shape=(2,), dtype=int), #agent coords
            "target": gym.spaces.Box(0, self.full_width - 1, shape=(2,), dtype=int), #target coords
            "facing": gym.spaces.Discrete(4) #facing direction
        })

        self._direction_vectors = {
            FacingDirection.RIGHT: np.array([1, 0]),
            FacingDirection.DOWN: np.array([0, 1]),
            FacingDirection.LEFT: np.array([-1, 0]),
            FacingDirection.UP: np.array([0, -1]),
        }

        self.action_space = spaces.Discrete(4)

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "facing": np.array([self._facing_direction.value])
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def step(self, action):
        prev_distance = np.linalg.norm(
            self._agent_location - self._target_location, ord=1
            )
        if action == 0:  # TURN_LEFT
            self._facing_direction = FacingDirection((self._facing_direction.value - 1) % 4)
        elif action == 1:  # TURN_RIGHT
            self._facing_direction = FacingDirection((self._facing_direction.value + 1) % 4)
        else:
            move = self._direction_vectors[self._facing_direction]
            if action == 3:  # MOVE_BACKWARD
                move = -move
            new_location = self._agent_location + move
            if (
                0 <= new_location[0] < self.full_width
                and 0 <= new_location[1] < self.size
            ):
                self._agent_location = new_location
                
        in_parked_car = (
            self._agent_location[1] in [0, 2] and # in a non-target square on top or bottom row
            not np.array_equal(self._agent_location, self._target_location)
        )

        terminated = np.array_equal(self._agent_location, self._target_location)
        new_distance = np.linalg.norm(
            self._agent_location - self._target_location, ord=1
        )
        shaping_reward = prev_distance - new_distance  # positive if agent got closer
        if terminated:
            reward = 1
        elif in_parked_car:
            reward = -5  # strong penalty for bumping into parked cars
        else:
            reward = shaping_reward - 1  # normal movement shaping reward

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
                (self.window_width, self.window_height)
            )

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        pix_square_size = self.tile_size

        agent_image = pygame.image.load("car.png")
        angle = {FacingDirection.RIGHT: 0, FacingDirection.DOWN: 90,
                 FacingDirection.LEFT: 180, FacingDirection.UP: 270}[self._facing_direction]
        agent_image = pygame.transform.rotate(
            pygame.transform.scale(agent_image, (int(pix_square_size), int(pix_square_size))),
            angle
        )

        parked_car_image = pygame.image.load("parked_car.png")
        parked_car_image = pygame.transform.scale(
            parked_car_image, (int(pix_square_size), int(pix_square_size))
        )

        for row in range(self.size):
            for col in range(self.full_width):
                if (row != 1) and not (row == self._target_location[1] and col == self._target_location[0]):
                    pos = np.array([col, row]) * pix_square_size
                    pos = pos.astype(int)
                    canvas.blit(parked_car_image, pos)

        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        agent_pos = (self._agent_location * pix_square_size).astype(int)
        canvas.blit(agent_image, agent_pos)

        for y in range(self.size + 1):
            pygame.draw.line(
                canvas, 0,
                (0, pix_square_size * y),
                (self.window_width, pix_square_size * y),
                width=3
            )

        for x in range(self.full_width + 1):
            pygame.draw.line(
                canvas, 0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_height),
                width=3
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            if self.clock is None:
                self.clock = pygame.time.Clock()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = np.array([0, 1], dtype=int)
        self._facing_direction = FacingDirection.RIGHT

        self._target_location = np.array([
            self.np_random.integers(0, self.full_width),
            self.np_random.choice([0, 2])
        ])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def flatten_obs(obs):
    return np.concatenate([obs["agent"], obs["target"], obs["facing"]]).astype(np.float32)

def train_parking_dqn(env, episodes=1000, gamma=0.99, lr=1e-3, batch_size=64,
                      epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                      target_update=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = 5
    n_actions = env.action_space.n

    policy_net = DQN(obs_dim, n_actions).to(device)
    target_net = DQN(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimiser = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(10000)

    epsilon = epsilon_start

    for episode in range(episodes):
        obs, _ = env.reset()
        state = torch.tensor(flatten_obs(obs), dtype=torch.float32).to(device)
        cum_reward = 0

        for t in count():
            if random.random() < epsilon:
                action = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.tensor(flatten_obs(next_obs), dtype=torch.float32).to(device)
            done = terminated or truncated
            memory.push(state, action, next_state, reward, done)
            state = next_state
            cum_reward += reward

            if done:
                break

            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                states = torch.stack(batch.state)
                actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(device)
                next_states = torch.stack(batch.next_state)
                rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(device)
                dones = torch.tensor(batch.done, dtype=torch.bool).unsqueeze(1).to(device)

                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    expected_q_values = rewards + gamma * max_next_q_values * (~dones)

                loss = F.mse_loss(q_values, expected_q_values)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode} - Total reward: {cum_reward} - Epsilon: {epsilon:.3f}")

if __name__ == "__main__":
    print("Starting Simulation")
    env = ParkingLot(render_mode="human", size=3)
    train_parking_dqn(env)

##test