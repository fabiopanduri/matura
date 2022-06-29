# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
import gym
import numpy as np

from dql.environment.cartpole_gym_env import CartpoleEnvDQL


class CartpoleEnvNEAT(CartpoleEnvDQL):
    """
    Provides the environment for the game Cartpole as implemented by the gym library to NEAT    """

    def __init__(self, plot=False, render=False):
        super().__init__(plot, render)
