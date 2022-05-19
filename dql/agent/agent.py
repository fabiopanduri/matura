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
    
    def store_transition(self, transition):
        '''
        Store a transition to replay memory (with corresponding time t of transition). Transition format: phi, action, reward, next_phi, terminal
        '''
        self.memory.append((time, transition))


    def sample(self, sample_size):
        '''
        Sample an amount of random uniform entries from replay memory
        '''
        return random.sample(self.memory, k=sample_size)
    

class DQLAgent:
    def __init__(self, env):
    '''
    Adjust these parameters as you wish
    '''
        # DQL Params
        self.env = env
        self.possible_actions = env.possible_actions
        self.memory_size = 10000
        self.memory = ReplayMemory(self.memory_size)
        self.eps = 0.01
        self.update_frequency = 100
        self.discount_factor = 0.01
        self.minibatch_size = 1

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


    def get_action(self, image):
        '''
        Select action based on eps-greedy policy based on Q network
        '''
        # With probability eps select random action of possible actions
        if random.random() < self.eps:
            action = random.choice(self.possible_actions)

        # Else select action which leads to max reward estimated by Q. 
        else:
            q_values = self.q_network.feed_forward(image)
            action = self.possible_actions[np.argmax(q_values)]

        return action

    def execute_action(self, action):
        '''
        Execute action in emulator and observe image and reward
        '''
        image, reward = self.env.step(action)
        return image, reward 

    def preprocessor(self, sequence):
        '''
        Preprocesses a sequence s returning a (typically smaller in size) preprocessed sequence phi.
        '''
        # Because for the moment no images are used, phi equals sequence and no preprocessing needs be done
        return sequence

    def gd_on_minibatch(self, minibatch):
        '''
        Perform stochastic gradient descent step on minibatch of stored transitions
        '''

        training_batch = []
        for time, transition in minibatch:
            phi, action, reward, next_phi, terminal = transition
            if terminal:
                # If episode terminates at next step, target reward equals current reward
                target_reward = reward
            else:
                # If episode doesn't terminate, target reward equals current reward + expected future reward
                target_q_values = self.target_q_network.feed_forward(next_phi)
                target_reward = reward + np.max(target_q_values)

            training_batch.append((phi, target_reward))

        self.gd_on_q_network(training_batch)


        approx_target_value = 69

    def learn(self, n_of_episodes, iterations):
        '''
        Perform Q Learning as described by Algorithm 1 in Mnih et al. 2015
        '''
        for episode in range(n_of_episodes):
            # Initial sequence is just the initial image
            sequence = (None, None, self.env.get_image())
            phi = self.preprocessor(sequence)
            for t in range(iterations):
                # Play one game step and observe new image and reward
                action = self.get_action(phi)
                next_image, reward, terminal = execute_action(action)
                next_sequence = (sequence, image, action, reward)
                next_phi = self.preprocessor(next_sequence)
                # Store transition in replay memory
                transition = (phi, action, reward, next_phi, terminal)
                self.memory.store(t, transition)

                # Sample minibatch from replay memory and train Q network on it
                minibatch = self.memory.sample(self.minibatch_size)
                self.experience_replay(minibatch)

                if t % self.update_frequency == 0:
                    # TODO: Do this with weight sync because python likes to copy by reference. Also implement storing networks to disk
                    self.target_q_network = self.q_network

    def play(self):
        return



def main():
    pass

if __name__ == '__main__': main()
