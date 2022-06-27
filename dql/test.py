# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
import os
import random
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from dql.agent.agent import DQLAgent
from dql.agent.agent import ReplayMemory
from dql.environment.pong_env import PongEnv
from dql.neural_network.neural_network import NeuralNetwork


def main():
    env = PongEnv()
    agt = DQLAgent(env, load_network_path='latest')

    agt.learn(100000)


if __name__ == '__main__':
    main()
