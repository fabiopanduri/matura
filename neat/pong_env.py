# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
# Pong game to be played by ML algorithms
import time

import numpy as np

from pong.pong import PADDLE_HEIGHT
from pong.pong import PongGame
from pong.pong import WINDOW_SIZE


class PongEnvNEAT:
    """
    Provides the environment for the game Pong to the NEAT Agent
    """

    def __init__(self, max_t=0, render=False, reward_system='v0'):
        '''
        Reset the game to initial state and return initial state
        '''
        self.game = PongGame(graphics_enabled=render)
        self.possible_actions = ['up', 'stay', 'down']
        self.state_size = len(self.make_observation())
        self.max_t = max_t

        self.done_fitness = 1
        self.reward_system = reward_system

    def nn_base_dimensions(self):
        return [self.state_size, len(self.possible_actions)]

    def make_observation(self):
        '''
        Return the current game's current internal state (relevant params)
        '''
        return (self.game.right_paddle.relative_y_position(), self.game.ball.relative_position()[1])

    def step(self, action, t):
        '''
        Do one game move with given action and return image, reward and wheter or not the game terminates
        '''
        # Get desired paddle movement from first (and only) entry of action tuple
        right_movement = action

        # For the time being, make the opponent unmoving
        left_movement = 'stay'

        # Perform one game tick. Store prev score to calculate reward
        prev_score = self.game.score.copy()

        # Note to self: bug was here (only terminated was collected and was thus list)
        terminated, right_paddle_collision = self.game.tick(
            left_movement,
            right_movement,
        )

        # Give out rewards according to reward system
        if self.reward_system == "v0":
            # Positive reward in winning, negative reward in losing frames
            if self.game.score[0] == prev_score[0] + 1:
                reward = -1
            elif self.game.score[1] == prev_score[1] + 1:
                reward = 1
            else:
                reward = 0

        elif self.reward_system == "v1":
            # Positive reward if agent hit the paddle
            if right_paddle_collision:
                reward = 1
            else:
                reward = 0

        elif self.reward_system == "v2":
            # Always give out positive rewards, except for winning/losing frames
            if self.game.score[0] == prev_score[0] + 1:
                reward = -1
            elif self.game.score[1] == prev_score[1] + 1:
                reward = 1
            else:
                reward = 0

        elif self.reward_system == "v3":
            # +1 if paddle height corresponds with ball height, -0.1 else
            if self.game.right_paddle.position[1] <= self.game.ball.position[1] <= self.game.right_paddle.position[1] + PADDLE_HEIGHT:
                reward = 1
            else:
                reward = -0.1
            # for this reward system we do not want the simulation to stop if a
            # point is gained
            if t == self.max_t - 1:
                terminated = True
            else:
                terminated = False

        elif self.reward_system == "v4":
            # +1 if ball is directly above paddle vertically,
            # -1 if it is as far away as the window is high,
            # exponential decay in between
            # reward = 1 - abs(self.game.ball.position[1] - (self.game.right_paddle.position[1] + PADDLE_HEIGHT / 2)) / WINDOW_SIZE[1]
            reward = 2**(-abs(self.game.ball.position[1] - (
                self.game.right_paddle.position[1] + PADDLE_HEIGHT / 2)) / 100)
            # print(reward)
            # for this reward system we do not want the simulation to stop if a
            # point is gained
            if t == self.max_t - 1:
                terminated = True
            else:
                terminated = False

        return self.make_observation(), reward, terminated

    def fitness(self, t, reward, alpha=1000):
        """
        Function to calculate the fitness of an individual based on time and reward he got
        """
        # v3-v4 uses reward directly as fitness
        if self.reward_system == 'v4' or self.reward_system == 'v3':
            return reward

        # v0-v2 use weighted reward depending on when the terminal state is reached
        if reward > 0:
            f = 1 + np.exp(-t/alpha)
        elif reward < 0:
            f = 1 - np.exp(-t/alpha)
        else:
            f = 1
        return f
