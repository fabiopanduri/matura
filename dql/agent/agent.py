# Copyright (C) 2022 Luis Hartmann and Fabio Panduri

# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.

import random
import numpy as np
from neural_network import NeuralNetwork

class ReplayMemory:
    def __init__(self, size):
        self.memory = [] 
    
    def store(self, experience_tuple):
        '''
        Store an experience tuple to memory
        '''
        self.memory.append(experience_tuple)


    def sample(self, sample_size):
        '''
        Sample an amount of random uniform experience tuples from replay memory to train DQN Agent
        '''
        return random.sample(self.memory, k=sample_size)


class DQLAgent:
    def __init__(self, env):
    '''
    Adjust these parameters as you wish
    '''
        # SGD Params
        self.minibatch_size = 32

        # DQL Params
        self.env = env
        self.possible_actions = env.possible_actions
        self.memory_size = 10000
        self.memory = ReplayMemory(self.memory_size)
        self.eps = 0.01
        self.update_frequency = 100
        self.discount_factor = 0.01

        # NN Params
        '''
        Neural Network.
        Input: Current observation image
        Output: Estimated reward for each possible action
        '''
        self.nn_dimensions = [5, 10, 10, len(self.possible_actions)]
        self.learning_rate = 0.1
        self.activation_functions = ['ReLU', 'ReLU', 'ReLU', 'ReLU']
        self.q_network = NeuralNetwork(self.nn_dimensions, self.learning_rate, self.activation_functions)
        self.target_q_network = self.q_network


    def get_action(self, observation):
        '''
        Select action based on eps-greedy policy based on Q network
        '''
        # With probability eps select random action of possible actions
        if random.random() < self.eps:
            action = random.choice(self.possible_actions)

        # Else select action which leads to max reward estimated by Q. 
        else:
            q_values = self.q_network.feed_forward(observation)
            action = np.argmax(q_values)

        return action

    def execute_action(self, action):
        '''
        Execute action in emulator and observe reward and new image
        '''
        image, reward = self.env.step(action)
        return image, reward

    def gd_on_q_network(self):
        pass

    def learn(self):
        pass

    def play(self):
        return



def main():
    pass

if __name__ == '__main__': main()
