# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of matura.
# matura is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# matura is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with matura. If not, see <https://www.gnu.org/licenses/>.
# Pong game to be played by ML algorithms
"""
Environment for the game pong to be used by DQL
"""
import matplotlib.pyplot as plt
import numpy as np

from pong.pong import PADDLE_HEIGHT
from pong.pong import PongGame


class PongEnvDQL:
    """
    Provide the environment for the game Pong to the DQL Agent.
    The Environment is always the right player.
    TODO: Write an abstraction which enables playing the left player.
    """

    def __init__(self, reward_system, alpha=1000, plot=False, render=True):
        '''
        Reset the game to initial state and return initial state
        '''
        self.game = PongGame(graphics_enabled=render)
        self.possible_actions = ['up', 'stay', 'down']
        self.state_size = len(self.make_observation())
        self.plot = plot
        self.alpha = alpha
        self.reward_system = reward_system
        # Needed for reward systems which calculate fitness as mean over rewards
        self.reward_hist = []

    def make_observation(self):
        '''
        Return the current game's current internal state (relevant params)
        '''
        return (self.game.right_paddle.relative_y_position(), self.game.ball.relative_position()[1])

    def step(self, action):
        '''
        Do one game move with given action and return image, reward and whether or not the game terminates
        '''
        # Get desired paddle movement from first (and only) entry of action tuple
        right_movement = action[0]

        # Opponent is unmoving
        left_movement = 'stay'

        # Perform one game tick. Store prev score to calculate reward
        prev_score = self.game.score.copy()
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
                reward = -5
            elif self.game.score[1] == prev_score[1] + 1:
                reward = 5
            else:
                reward = 1

        elif self.reward_system == "v3":
            # +1 if paddle height corresponds with ball height, 0 else
            if self.game.right_paddle.position[1] <= self.game.ball.position[1] <= self.game.right_paddle.position[1] + PADDLE_HEIGHT:
                reward = 1
            else:
                reward = 0
            self.reward_hist.append(reward)

        return self.make_observation(), reward, terminated

    def fitness(self, t, reward):
        """
        Function to calculate the fitness of an individual based on time and reward he got
        """
        # v3 defines fitness as average over rewards
        if self.reward_system == 'v3':
            f = sum(self.reward_hist) / len(self.reward_hist)
            return f

        # v0-v2 use weighted reward depending on when the terminal state is reached
        if reward > 0:
            f = 1 + np.exp(-t/self.alpha)
        elif reward < 0:
            f = 1 - np.exp(-t/self.alpha)
        else:
            f = 1
        return f

    def terminate_episode(self):
        '''
        Function to be called after an episode (iteration of the game) ends. 
        '''
        self.game.reset()
