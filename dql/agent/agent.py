# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of matura.
# matura is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# matura is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with matura. If not, see <https://www.gnu.org/licenses/>.
"""
DQL Agent class containing main DQL algorithm
and ReplayMemory class
"""
import datetime
import json
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
    '''
    Implement replay memory as a tuple with a fixed size
    '''

    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = deque()

    def size(self):
        return len(self.memory)

    def store(self, transition):
        '''
        Store a transition to replay memory (with corresponding time t of transition). Transition format: phi, action, reward, next_phi, done
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
    def __init__(self,
                 env,
                 memory_size,
                 discount_factor,
                 minibatch_size,
                 learning_rate,
                 eps_annealing_rate,
                 done_eps,
                 target_nn_update_freq,
                 max_simulation_time,
                 load_network_path,
                 live_plot_freq,
                 live_plot,
                 nn_save_freq,
                 save_data=False,
                 game="",
                 ):

        # Hyperparameters are defined in the config.py file
        self.env = env
        self.memory_size = memory_size
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.eps_annealing_rate = eps_annealing_rate
        self.done_eps = done_eps
        self.target_nn_update_freq = target_nn_update_freq
        self.max_simulation_time = max_simulation_time
        self.live_plot = live_plot
        self.live_plot_freq = live_plot_freq
        self.nn_save_freq = nn_save_freq
        self.save_data = save_data
        self.game = game

        self.possible_actions = self.env.possible_actions
        self.memory = ReplayMemory(self.memory_size)
        self.total_step = 0

        # Neural Network input: game state; output: expected future discounted reward
        self.nn_dimensions = [self.env.state_size,
                              24, 24, len(self.possible_actions)]
        self.activation_functions = ['ReLU', 'ReLU', 'ReLU', 'linear']

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

        # Initalize target network
        self.update_target_network()

        self.performance_hist = []
        self.time_hist = []

    def update_target_network(self):
        self.target_q_network = NeuralNetwork(
            self.nn_dimensions, self.learning_rate, self.activation_functions, self.q_network.weights, self.q_network.biases)

    def get_eps(self, t):
        '''
        Get Epsilon (exploration rate). 
        '''
        # Exponential decay:
        eps = max(self.done_eps, self.eps_annealing_rate**t)
        return eps

    def get_action(self, state):
        '''
        Select action (movement) based on eps-greedy policy based on Q network
        '''
        # With probability eps select random movement of possible movements
        if random.random() < self.get_eps(self.total_step):
            q_values = 'random action'
            movement = random.choice(self.possible_actions)

        # Else select action which leads to max reward estimated by Q.
        else:
            q_values = self.q_network.feed_forward(state)
            movement = self.possible_actions[np.argmax(q_values)]

        # Movement is a list because an action might include multiple values for
        # different envs
        return [movement]

    def execute_action(self, action):
        '''
        Execute action in emulator and observe state and reward and "done" indicating if a step is terminal
        '''
        state, reward, done = self.env.step(action)
        return state, reward, done

    def preprocessor(self, state):
        '''
        Preprocesses a state s returning a (typically smaller in size) preprocessed state phi.
        Placeholder function in case the implementation should be expanded to work on images
        '''
        # Because no images are used, phi = state and no preprocessing needs to be done
        return state

    def replay(self):
        '''
        Perform experience replay on minibatch of transitions from memory. Update network with stochastic gradient descent.
        '''
        # Don't perform replay if memory is too small
        if self.minibatch_size > self.memory.size():
            return

        minibatch = self.memory.sample(self.minibatch_size)
        training_batch = []  # To perform SGD on
        for transition in minibatch:
            phi, action, reward, next_phi, done = transition

            # Initially, target = network prediction
            target_rewards = self.q_network.feed_forward(phi)
            # If episode terminates at next step (done=True),
            # reward = current reward for the taken action.
            taken = self.possible_actions.index(action[0])
            target_rewards[taken] = reward

            if not done:
                # If episode doesn't terminate, add the estimated rewards for each future action
                target_rewards[taken] += self.discount_factor * np.max(
                    self.target_q_network.feed_forward(next_phi))

            training_batch.append((np.array(phi), target_rewards))

        self.q_network.stochastic_gradient_descent(training_batch)

    def learn(self, n_of_episodes):
        '''
        Perform Q Learning as described by Algorithm 1 in Mnih et al. 2015
        and implemented in Keon 2017
        '''
        for episode in range(n_of_episodes):
            done = False
            state = self.env.make_observation()
            phi = self.preprocessor(state)

            episode_step = 0
            t_0 = time.perf_counter()
            # Play until episode is done or max simulation time exceeded
            while not done and episode_step < self.max_simulation_time:
                # Play one frame and observe new state and reward
                action = self.get_action(phi)
                state, reward, done = self.execute_action(action)
                next_phi = self.preprocessor(state)

                # Perform experience replay
                transition = (phi, action, reward, next_phi, done)
                self.memory.store(transition)
                self.replay()

                # Update target network every C steps
                if self.total_step % self.target_nn_update_freq == 0:
                    self.update_target_network()

                # Save network to disk
                if (self.total_step + 1) % self.nn_save_freq == 0:
                    self.q_network.save_network()

                phi = next_phi
                episode_step += 1
                self.total_step += 1

            # Track performance
            t = time.perf_counter() - t_0
            self.time_hist.append(t)
            f = self.env.fitness(episode_step, reward)
            self.performance_hist.append(f)

            # information about the algorithm
            print(f'[INFO] Episode {episode} done. Took {t:.2f} s')
            print(f'[INFO] Fitness: {f}\n')

            if self.live_plot:
                self.plot_performance()

            self.env.terminate_episode()

        # Save training data
        if self.save_data:
            data = {
                "time": datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                "episodes": n_of_episodes,
                "performance history": self.performance_hist,
                "time history": self.time_hist,
                "hyperparameters": {
                    "memory size": self.memory_size,
                    "discount factor": self.discount_factor,
                    "minibatch size": self.minibatch_size,
                    "learning rate": self.learning_rate,
                    "epsilon decay": self.eps_annealing_rate,
                    "final epsilon": self.done_eps,
                    "target neural network update freq": self.target_nn_update_freq,
                    "live plot freq": self.live_plot_freq,
                    "live plot": self.live_plot,
                    "neural network save freq": self.nn_save_freq,
                },
                "game": self.game
            }

            file_name = (
                f"DQL-epoch-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
            )

            with open(f"DQL_epochs_saves/{file_name}", "w") as f:
                f.write(json.dumps(data, indent=4))

            print(
                f"[INFO] Saved epoch to DQL_epochs_saves/{file_name}.")

    def plot_performance(self):
        """
        Plot the performance of the game up to now
        """

        y = self.performance_hist
        x = list(range(len(y)))

        plt.plot(x, y)
        plt.show(block=False)
        plt.pause(0.001)
