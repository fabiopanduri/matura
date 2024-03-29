# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of matura.
# matura is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# matura is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with matura. If not, see <https://www.gnu.org/licenses/>.
# Pong game to be played by ML algorithms
"""
Environment for the game cartpole to be used by DQL
"""
import time

import gym
import numpy as np


class CartpoleEnvDQL:
    """
    Provide the environment for the game CartPole to the DQL Agent
    """

    def __init__(self, reward_system, alpha=1000, render=True):
        '''
        Reset the game to initial state and return initial state
        '''
        self.gym_env = gym.make('CartPole-v1')
        # Note actions are defined through self.gym_env.action_space == Discrete(2)
        self.possible_actions = [0, 1]
        self.state_size = len(self.make_observation())
        self.render = render
        self.alpha = alpha

    def fitness(self, t, reward):
        """
        Function to calculate the fitness of an individual based on time and reward he got
        """
        return t

    def make_observation(self):
        '''
        Return the current game's current internal state (relevant params)
        '''
        # Reset gives back the gym env state
        return self.gym_env.reset()

    def step(self, action):
        '''
        Do one game move with given action and return image, reward and wheter or not the game terminates
        '''
        observation, reward, done, info = self.gym_env.step(action[0])
        if self.render:
            self.gym_env.render()
        return observation, reward, done

    def terminate_episode(self):
        '''
        Function to be called after an episode (iteration of the game) ends
        '''
        pass

