# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
import random
import sys
from collections import deque

import numpy as np

from dql.neural_network.neural_network import NeuralNetwork
from pong.pong import PongEnv


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
    def __init__(self, env):
        '''
        Adjust these parameters as you wish
        '''
        # DQL Params
        self.env = env
        self.possible_actions = env.possible_actions
        self.memory_size = 10000
        self.memory = ReplayMemory(self.memory_size)
        self.update_frequency = 100
        self.discount_factor = 0.01
        self.minibatch_size = 32
        self.total_step = 0

        # NN Params
        '''
        Neural Network.
        Input: Current state 
        Output: Estimated reward for each possible action
        '''
        self.nn_dimensions = [self.env.state_size,
                              10, 10, len(self.possible_actions)]
        self.learning_rate = 0.1
        self.activation_functions = ['sigmoid' for _ in range(4)]
        self.q_network = NeuralNetwork(
            self.nn_dimensions, self.learning_rate, self.activation_functions)
        self.q_network.initialize_network()
        self.update_target_network()
        self.save_frequency = 10000

    def update_target_network(self):
        self.target_q_network = NeuralNetwork(
            self.nn_dimensions, self.learning_rate, self.activation_functions, self.q_network.weights, self.q_network.biases)

    def get_eps(self, step, terminal_eps=0.01):
        '''
        Get Epsilon (exploration rate). Linearly adjusted from 1.0 to terminal_eps.
        '''
        # y = -(1-terminal_eps)/1'000'000x + 1.0
        eps = - (0.8 - terminal_eps) / 100000 * step + 1.0
        return eps

    def get_action(self, state):
        '''
        Select action (movement) based on eps-greedy policy based on Q network
        '''
        # With probability eps select random movement of possible movements
        if random.random() < self.get_eps(self.total_step):
            movement = random.choice(self.possible_actions)

        # Else select action which leads to max reward estimated by Q.
        else:
            q_values = self.q_network.feed_forward(state)
            print(q_values)
            movement = self.possible_actions[np.argmax(q_values)]

        return [movement]

    def execute_action(self, action):
        '''
        Execute action in emulator and observe state and reward, also score for determining if state is terminal
        '''
        state, reward, score = self.env.step(action)
        return state, reward, score

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
            # self.q_network.print_network()
            # If episode terminates at next step, reward = current reward for the taken action.
            taken = self.possible_actions.index(action[0])
            target_rewards[taken] = reward

            if not terminal:
                # If episode doesn't terminate, add the estimated rewards for each future action
                target_q_values = self.target_q_network.feed_forward(next_phi)
                for i in range(len(target_rewards)):
                    target_rewards[i] += target_q_values[i]

            training_batch.append((np.array(phi), target_rewards))

        self.q_network.stochastic_gradient_descent(training_batch)

    def learn(self, n_of_episodes):
        '''
        Perform Q Learning as described by Algorithm 1 in Mnih et al. 2015
        '''
        for episode in range(n_of_episodes):
            terminal = False
            score = self.env.score
            state = self.env.make_observation()
            phi = self.preprocessor(state)

            step = 0
            # Play until terminal state/frame is reached
            while not terminal:
                # Play one frame and observe new state and reward
                action = self.get_action(phi)
                state, reward, next_score = self.execute_action(action)
                next_phi = self.preprocessor(state)

                # Check if episode terminates, it does when the score updates
                terminal = score != next_score

                transition = (phi, action, reward, next_phi, terminal)
                self.memory.store(transition)

                self.replay()

                if step % self.update_frequency == 0:
                    self.q_network.print_network()
                    self.target_q_network = self.q_network
                    self.update_target_network()

                if step % self.step_frequency == 0:
                    print(self.total_step)
                    self.q_network.save_network()

                # Roll over all variables
                score, phi = [i for i in next_score], next_phi
                step += 1
                self.total_step += 1

    def play(self):
        return


def main():
    env = PongEnv()
    agt = DQLAgent(env)

    agt.learn(1000)


if __name__ == '__main__':
    main()
