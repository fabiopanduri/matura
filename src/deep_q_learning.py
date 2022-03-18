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
        This method feeds the current state through the DQN and returns the probabilites of the next actions.
        '''

        pass



def main():
    pass

if __name__ == '__main__': main()
