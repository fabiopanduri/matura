# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
# Pong game to be played by ML algorithms
import math
from typing import List
from typing import Tuple

import numpy as np


def circle_corner_bounce(
    corner: Tuple[int, int], circle: Tuple[int, int, int], circle_velocity
):
    """
    Handle bounce of a circle with a single point
    """

    # Deconstruct velocity vector into component parallel and component perpendicular to touching radius
    radius_vector = np.array([corner[0] - circle[0], corner[1] - circle[1]])
    radius_normal_vector = np.array([-radius_vector[1], radius_vector[0]])

    # solve equation velocity = u * (radius_vector) + v * (radius_normal_vector) <=> A * scalars = velocity
    A = np.array(
        [
            [radius_vector[0], radius_normal_vector[0]],
            [radius_vector[1], radius_normal_vector[1]],
        ]
    )

    scalars = np.linalg.solve(A, circle_velocity)
    parallel_velocity = scalars[0] * radius_vector
    perpendicular_velocity = scalars[1] * radius_normal_vector

    # When colliding a circle with a point, the resulting velocity = -1 * (velocity_parallel_to_radius) + (velocity_perpendicular_to_radius)
    return -1 * (parallel_velocity) + (perpendicular_velocity)


def rect_circle_collision(
    rectangle: Tuple[int, int, int, int], circle: Tuple[int, int, int], circle_velocity
):
    # TODO: This code is hella DONOTREPEATYOURSELF...
    """
    Collide a moving circle with a rectangle. Parse rectangle as top left and bottom right point, circle as centre and radius.
    """

    # There are 8 possibilites where the ball can be relative to the rectangle.
    # Four are facing each side of the rectangle
    # Four are closest to each corner

    # Left side
    if circle[0] < rectangle[0] and rectangle[1] < circle[1] < rectangle[3]:
        # Collision?
        if circle[0] + circle[2] >= rectangle[0]:
            return np.multiply(circle_velocity, np.array([-1, 1])), True

    # Bottom side
    if circle[1] > rectangle[2] and rectangle[0] < circle[0] < rectangle[2]:
        # Collision?
        if circle[1] - circle[2] <= rectangle[1]:
            return np.multiply(circle_velocity, np.array([1, -1])), True

    # Right side
    if circle[0] > rectangle[2] and rectangle[1] < circle[1] < rectangle[3]:
        # Collision?
        if circle[0] - circle[2] <= rectangle[2]:
            return np.multiply(circle_velocity, np.array([-1, 1])), True

    # Top side
    if circle[1] < rectangle[1] and rectangle[0] < circle[0] < rectangle[2]:
        # Collision?
        if circle[1] + circle[2] >= rectangle[1]:
            return np.multiply(circle_velocity, np.array([1, -1])), True

    # If the ball is not facing any sides, check if it makes contact with any corner
    # Top left corner
    if (rectangle[0] - circle[0]) ** 2 + (rectangle[1] - circle[1]) ** 2 <= circle[
        2
    ] ** 2:
        return circle_corner_bounce(
            (rectangle[0], rectangle[1]), circle, circle_velocity
        ), True

    # Bottom left corner
    if (rectangle[0] - circle[0]) ** 2 + (rectangle[3] - circle[1]) ** 2 <= circle[
        2
    ] ** 2:
        return circle_corner_bounce(
            (rectangle[0], rectangle[3]), circle, circle_velocity
        ), True

    # Bottom right corner
    if (rectangle[2] - circle[0]) ** 2 + (rectangle[3] - circle[1]) ** 2 <= circle[
        2
    ] ** 2:
        return circle_corner_bounce(
            (rectangle[2], rectangle[3]), circle, circle_velocity
        ), True

    # Top right corner
    if (rectangle[2] - circle[0]) ** 2 + (rectangle[1] - circle[1]) ** 2 <= circle[
        2
    ] ** 2:
        return circle_corner_bounce(
            (rectangle[2], rectangle[1]), circle, circle_velocity
        ), True

    # If none of the above options are applicable, the circle didn't collide, so just return
    # the input velocity
    return circle_velocity, False
