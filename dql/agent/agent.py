# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
import os
import random
import sys
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from dql.environment.cartpole_gym_env import CartpoleEnvDQL
from dql.environment.pong_env import PongEnvDQL
from dql.neural_network.neural_network import NeuralNetwork


class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = deque()

    def size(self):
        return len(self.memory)

    def store(self, transition):
        '''
        Store a transition to replay memory (with corresponding time t of transition). Transition format: phi, action, reward, next_phi, terminal
        If memory is full, delete oldest transition.
        '''
        self.memory.append(transition)
        while self.size() > self.max_size:
            self.memory.popleft()

    def sample(self, sample_size):
        '''
        Sample an amount of random uniform entries from replay memory
        '''
        return random.sample(self.memory, sample_size)


class DQLAgent:
    def __init__(self, env, load_network_path=None):
        '''
        Adjust these parameters as you wish
        '''
        self.env = env
        self.possible_actions = env.possible_actions
        self.memory_size = 10000
        self.memory = ReplayMemory(self.memory_size)
        self.update_frequency = 100
        self.save_frequency = 10000
        self.discount_factor = 0.99
        self.minibatch_size = 32
        self.total_step = 0
        self.learning_rate = 0.1

        # Neural Network.
        # Input: Current state
        # Output: Estimated reward for each possible action
        self.nn_dimensions = [self.env.state_size,
                              5, len(self.possible_actions)]
        self.activation_functions = ['ReLU', 'ReLU', 'linear']

        # Allows for loading of previously trained q_networks from files
        if load_network_path == None:
            self.q_network = NeuralNetwork(
                self.nn_dimensions, self.learning_rate, self.activation_functions)
            self.q_network.initialize_network(value_range=(-1, 1))

        elif load_network_path == 'latest':
            latest = sorted(list(os.listdir('NN_saves')))[-1]
            if latest != 'z':
                self.q_network = NeuralNetwork.load_network(
                    f'NN_saves/{latest}')
            else:
                self.q_network = NeuralNetwork(
                    self.nn_dimensions, self.learning_rate, self.activation_functions)
                self.q_network.initialize_network(value_range=(-1, 1))
        else:
            self.q_network = NeuralNetwork.load_network(load_network_path)

        self.update_target_network()

        self.fitness_hist = []
        self.time_hist = []

    def update_target_network(self):
        self.target_q_network = NeuralNetwork(
            self.nn_dimensions, self.learning_rate, self.activation_functions, self.q_network.weights, self.q_network.biases)

    def get_eps(self, step, terminal_eps=0.1):
        '''
        Get Epsilon (exploration rate). 
        '''
        # Linear:
        # eps = - (1.0 - terminal_eps) / 100000 * step + 1.0
        # Exp decay:
        #eps = max(terminal_eps, 1.0 * (2 ** (-step / 5000)))
        # Lin decay:
        eps = max(terminal_eps, -0.001 * step + 1)
        return eps

    def get_action(self, state):
        '''
        Select action (movement) based on eps-greedy policy based on Q network
        '''
        # With probability eps select random movement of possible movements
        if random.random() < self.get_eps(self.total_step):
            movement = random.choice(self.possible_actions)
            q_values = 'random action'

        # Else select action which leads to max reward estimated by Q.
        else:
            q_values = self.q_network.feed_forward(state)
            movement = self.possible_actions[np.argmax(q_values)]

        if self.total_step % self.update_frequency == 0:
            print(self.total_step, q_values, "Max: ",
                  self.possible_actions[np.argmax(q_values)])
            if self.env.plot == True:
                self.env.score_hist.append(self.env.game.score.copy())
                self.env.plot_score()

        return [movement]

    def execute_action(self, action):
        '''
        Execute action in emulator and observe state and reward, also score for determining if state is terminal
        '''
        state, reward, terminated = self.env.step(action)
        return state, reward, terminated

    def preprocessor(self, state):
        '''
        Preprocesses a state s returning a (typically smaller in size) preprocessed state phi.
        '''
        # Because for the moment no actual images are used, phi equals state and no preprocessing needs be done
        return state

    def replay(self):
        '''
        Perform experience replay on minibatch of transitions from memory. Update network with stochastic gradient descent.
        '''
        # Dont perform replay if memory is too small
        if self.minibatch_size > self.memory.size():
            return

        minibatch = self.memory.sample(self.minibatch_size)
        training_batch = []
        for transition in minibatch:
            phi, action, reward, next_phi, terminal = transition

            # Initially, target = network prediction
            target_rewards = self.q_network.feed_forward(phi)
            # If episode terminates at next step (terminal=True), reward = current reward for the taken action.
            taken = self.possible_actions.index(action[0])
            target_rewards[taken] = reward

            if not terminal:
                # If episode doesn't terminate, add the estimated rewards for each future action
                target_rewards[taken] += self.discount_factor * np.max(
                    self.target_q_network.feed_forward(next_phi))

            training_batch.append((np.array(phi), target_rewards))

        self.q_network.stochastic_gradient_descent(training_batch)

    def learn(self, n_of_episodes):
        '''
        Perform Q Learning as described by Algorithm 1 in Mnih et al. 2015
        '''
        for episode in range(n_of_episodes):
            terminated = False
            state = self.env.make_observation()
            phi = self.preprocessor(state)

            step = 0
            # Play until terminal state/frame is reached
            t_0 = time.perf_counter()
            while not terminated:
                # Play one frame and observe new state and reward
                action = self.get_action(phi)
                state, reward, terminated = self.execute_action(action)
                # print(state, reward)
                next_phi = self.preprocessor(state)

                transition = (phi, action, reward, next_phi, terminated)
                self.memory.store(transition)

                self.replay()

                if step % self.update_frequency == 0:
                    self.update_target_network()

                # The +1 is that it doesn't always save on start -> less clutter
                if (self.total_step + 1) % self.save_frequency == 0:
                    self.q_network.save_network()

                # Roll over all variables
                phi = next_phi
                step += 1
                self.total_step += 1

            t = time.perf_counter() - t_0
            self.time_hist.append(t)

            fitness = self.env.fitness(step, reward)
            self.fitness_hist.append(fitness)

    def play(self):
        return


def main():
    env = PongEnvDQL(plot=True)
    agt = DQLAgent(env)

    agt.learn(100000)


if __name__ == '__main__':
    main()
