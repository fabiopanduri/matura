# Copyright (C) 2022 Luis Hartmann and Fabio Panduri

# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.

from neural_network import NeuralNetwork

from typing import Tuple

Position = Tuple[int, int]


class PongState:
    def __init__(self, ball_position: Position, ball_velocity: 'numpy_array', agent_position: Position) -> None:
        self.ball_position = ball_position
        self.ball_velocity = ball_velocity
        self.agent_position = agent_position

    def convert_state(self) -> 'numpy_array':
        '''
        This method converts the state object into a numpy array that is feedable to the Neural Network.
        '''

        pass


class DeepQLearning:
    def __init__(self, DQN: NeuralNetwork, s_0: 'State') -> None:
        self.DQN = DQN
        self.state = s_0


    def act(self, state: 'State') -> 'numpy_array':
        '''
        This method feeds the current state through the DQL and returns the probabilities of the next actions.
        '''

        pass



def main():
    pass

if __name__ == '__main__': main()
