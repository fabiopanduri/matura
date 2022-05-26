# Copyright (C) 2022 Luis Hartmann and Fabio Panduri

# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.

import sys
sys.path.insert(0,'/home/lnapkin/Documents/maturaarbeit_code')
import random
import numpy as np
from dql.neural_network.neural_network import NeuralNetwork
from pong.pong import PongEnv

class ReplayMemory:
    # TODO: Implement max size for Replay Memory
    def __init__(self, size):
        self.memory_size = size
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
        self.minibatch_size = 32

        # NN Params
        '''
        Neural Network.
        Input: Current state 
        Output: Estimated reward for each possible action
        '''
        self.nn_dimensions = [self.env.state_size, 10, 10, len(self.possible_actions)]
        self.learning_rate = 0.1
        self.activation_functions = ['ReLU', 'ReLU', 'ReLU', 'ReLU']
        self.q_network = NeuralNetwork(self.nn_dimensions, self.learning_rate, self.activation_functions)
        self.target_q_network = self.q_network


    def get_action(self, state):
        '''
        Select action based on eps-greedy policy based on Q network
        '''
        # With probability eps select random action of possible actions
        if random.random() < self.eps:
            action = random.choice(self.possible_actions)

        # Else select action which leads to max reward estimated by Q. 
        else:
            print(state)
            q_values = self.q_network.feed_forward(state)
            action = self.possible_actions[np.argmax(q_values)]

        return action

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

    def gd_on_minibatch(self, minibatch):
        '''
        Perform stochastic gradient descent step on minibatch of stored transitions
        '''
        training_batch = []
        for transition in minibatch:
            phi, action, reward, next_phi, terminal = transition

            # Initially, target = network prediction
            target_rewards = self.q_network.feed_forward(phi)
            # If episode terminates at next step, reward = current reward for the taken action. 
            taken = self.possible_actions.index(action)
            target_rewards[taken] = reward

            if not terminal:
                # If episode doesn't terminate, add the estimated rewards for each future action
                target_q_value = self.target_q_network.feed_forward(next_phi)
                for i in range(len(target_rewards)):
                    target_rewards[i] += target_q_values[i]

            training_batch.append((phi, target_rewards))

        self.gd_on_q_network(training_batch)

    def learn(self, n_of_episodes):
        '''
        Perform Q Learning as described by Algorithm 1 in Mnih et al. 2015
        '''
        for episode in range(n_of_episodes):
            terminal = False
            score = self.env.score
            # Initalize state 
            state = self.env.make_observation()
            phi = self.preprocessor(state)

            # Play until terminal state/frame is reached
            while not terminal:
                # Play one frame and observe new state and reward
                action = self.get_action(phi)
                state, reward, next_score = execute_action(action)
                next_phi = self.preprocessor(state)


                # Check if episode terminates, it does when the score updates
                terminal = score != next_score

                transition = (phi, action, reward, next_phi, terminal)
                self.memory.store(transition)

                minibatch = self.memory.sample(self.minibatch_size)
                self.experience_replay(minibatch)

                if t % self.update_frequency == 0:
                    # TODO: Do this with weight sync because python likes to copy by reference. Also implement storing networks to disk
                    self.target_q_network = self.q_network.copy()

                # Roll over all variables
                state, score, phi = next_state, next_score, next_phi

    def play(self):
        return



def main():
    env = PongEnv()
    agt = DQLAgent(env)
    
    agt.learn(10)

if __name__ == '__main__': main()
