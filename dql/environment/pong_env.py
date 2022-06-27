# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
# Pong game to be played by ML algorithms
import matplotlib.pyplot as plt
import numpy as np

from pong.pong import PongGame


class PongEnvDQL:
    """
    Provide the environment for the game Pong to the DQL Agent
    """

    def __init__(self, plot=False, render=True):
        '''
        Reset the game to initial state and return initial state
        '''
        self.game = PongGame(graphics_enabled=render)
        self.possible_actions = ['up', 'stay', 'down']
        self.state_size = len(self.make_observation())
        self.plot = plot

    def current_performance(self):
        '''
        Return current game performance (score ratio right/left)
        '''
        if self.game.score[0] == 0:
            return 0
        return self.game.score[1] / self.game.score[0]

    def make_observation(self):
        '''
        Return the current game's current internal state (relevant params)
        '''
        return (self.game.right_paddle.relative_y_position(), self.game.ball.relative_position()[1])

    def step(self, action):
        '''
        Do one game move with given action and return image, reward and wheter or not the game terminates
        '''
        # Get desired paddle movement from first (and only) entry of action tuple
        right_movement = action[0]

        # For the time being, make the opponent unmoving
        left_movement = 'stay'

        # Perform one game tick. Store prev score to calculate reward
        prev_score = self.game.score.copy()
        terminated = self.game.tick(
            left_movement,
            right_movement,
        )

        if self.game.score[0] == prev_score[0] + 1:
            # Negative reward if opponent gets a point
            reward = -1
        elif self.game.score[1] == prev_score[1] + 1:
            # Positive reward if agent gets a point
            reward = 1
        else:
            # Slight negative / zero reward if no point made
            reward = 0

        return self.make_observation(), reward, terminated

    def fitness(self, t, reward, alpha=1000):
        """
        Function to calculate the fitness of an individual based on time and reward he got
        """
        # weighted reward depending on when the terminal state is reached
        if reward > 0:
            f = 1 + np.exp(-t/alpha)
        elif reward < 0:
            f = 1 - np.exp(-t/alpha)
        else:
            f = 1
        return f

    def terminate_episode(self):
        '''
        Function to be called after an episode (iteration of the game) ends. (No purpose in pong but needed for consistency)
        '''
        return


def main():
    env = PongEnv()
    clock = pygame.time.Clock()
    while True:
        # print("Step: ", env.step(['stay']), "Score: ", env.score)
        clock.tick(FPS_LIMIT)
