# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of matura.
# matura is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# matura is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with matura. If not, see <https://www.gnu.org/licenses/>.
# Pong game to be played by ML algorithms
"""
The game pong
"""
import math
import random
import time
from typing import List
from typing import Tuple

import numpy as np
import pygame

from pong.geometry import *

# Adjust as needed
# (0|0) is on the top left. x-Axis is right, y-Axis down
WINDOW_SIZE = (800, 500)
BALL_RADIUS = 12
BALL_SPEED = 4
PADDLE_SPEED = 10
PADDLE_WIDTH = 20
PADDLE_HEIGHT = 100
COLOR_BACKGROUND = (0, 0, 0)
COLOR_FOREGROUND = (100, 100, 100)
COLOR_SPRITE = (255, 255, 255)
FPS_LIMIT = 60


class Paddle:
    def __init__(self, side):
        self.side = side if side in ["left", "right"] else "left"
        self.position = [
            0,
            (WINDOW_SIZE[1] - PADDLE_HEIGHT) / 2,
        ]  # Start paddle in the middle (vertically) of according side
        if self.side == "left":
            self.position[0] = 0
        elif self.side == "right":
            self.position[0] = WINDOW_SIZE[0] - PADDLE_WIDTH

    def move(self, direction):
        """
        Handle movement and top / bottom Collision
        """
        if direction == "stay":
            pass

        elif direction == "up":
            self.position[1] = max(self.position[1] - PADDLE_SPEED, 0)

        elif direction == "down":
            self.position[1] = min(
                self.position[1] + PADDLE_SPEED, WINDOW_SIZE[1] - PADDLE_HEIGHT
            )

    def relative_y_position(self):
        """
        Returns a value from 0 to 1 indicating how far the paddle has travelled
        """

        # Divide the absolute y-position by the window height - paddle height.
        # Because the paddle can only move as far down as it's height allows
        # The subraction is that resizing the paddle doesn't change the values
        return self.position[1] / (WINDOW_SIZE[1] - PADDLE_HEIGHT)


class Ball:
    def __init__(self):
        self.position = np.array([WINDOW_SIZE[0] / 2, WINDOW_SIZE[1] / 2])

        # Ball should start at a random angle. Achieved by setting random starting y-Velocity
        # Multiply by 0.9 that ball doesn't start with only vertical movement
        random_y_velocity = random.uniform(-0.9 * BALL_SPEED, 0.9 * BALL_SPEED)

        # norm of velocity must = BALL_SPEED
        self.velocity = np.array(
            [math.sqrt(BALL_SPEED**2 - random_y_velocity**2),
             random_y_velocity]
        )

    def reset(self):
        self.__init__()

    def border_collision(self):
        side_collision = ''
        if (self.position[0] - BALL_RADIUS) <= 0:
            # Collision with left wall
            self.reset()
            side_collision += 'left'

        elif (self.position[0] + BALL_RADIUS) >= WINDOW_SIZE[0]:
            # Collision with right wall
            self.reset()
            side_collision += 'right'

        if (self.position[1] - BALL_RADIUS) <= 0 or (
            self.position[1] + BALL_RADIUS
        ) >= WINDOW_SIZE[1]:
            # Collision with top or bottom wall
            self.velocity = np.multiply(self.velocity, np.array([1, -1]))

        return side_collision

    def paddle_collision(self, left_paddle, right_paddle):
        # Apply rectangle-circle collision function to ball with both paddles
        self.velocity, left_paddle_collision = rect_circle_collision(
            (
                left_paddle.position[0],
                left_paddle.position[1],
                left_paddle.position[0] + PADDLE_WIDTH,
                left_paddle.position[1] + PADDLE_HEIGHT,
            ),
            (self.position[0], self.position[1], BALL_RADIUS),
            self.velocity,
        )
        self.velocity, right_paddle_collision = rect_circle_collision(
            (
                right_paddle.position[0],
                right_paddle.position[1],
                right_paddle.position[0] + PADDLE_WIDTH,
                right_paddle.position[1] + PADDLE_HEIGHT,
            ),
            (self.position[0], self.position[1], BALL_RADIUS),
            self.velocity,
        )
        return left_paddle_collision, right_paddle_collision

    def update(self, left_paddle, right_paddle):
        '''
        Check for collision with walls (side and top/bottom) and paddles.
        Move ball according to velocity.
        Side collisions are returned to calculate the score
        '''
        side_collision = self.border_collision()
        left_paddle_collision, right_paddle_collision = self.paddle_collision(
            left_paddle, right_paddle)

        self.position += self.velocity
        return side_collision, right_paddle_collision

    def relative_position(self):
        """
        Return the balls position in the playing field as two floats from 0 to 1
        (relative horizontal and vertical position)
        """

        # Window acessible to ball = window size - 2 * ball radius,
        # as on each side the radius limits how close ball can approach side, so divide by window size - 2 * radius
        #
        # And ball has travelled 0 % when it is "ball radius" away from border, so subtract this from ball position
        return (self.position[0] - BALL_RADIUS) / (WINDOW_SIZE[0] - 2 * BALL_RADIUS), (
            self.position[1] - BALL_RADIUS
        ) / (WINDOW_SIZE[1] - 2 * BALL_RADIUS)


class PaddleSprite(pygame.sprite.Sprite):
    """
    Display paddles as pygame sprites
    """

    def __init__(self):
        super().__init__()

        self.image = pygame.Surface([PADDLE_WIDTH, PADDLE_HEIGHT])
        self.image.fill(COLOR_BACKGROUND)
        self.image.set_colorkey(COLOR_BACKGROUND)

        pygame.draw.rect(self.image, COLOR_SPRITE, [
                         0, 0, PADDLE_WIDTH, PADDLE_HEIGHT])

        self.rect = self.image.get_rect()

    def update(self, paddle_position):
        self.rect.x = paddle_position[0]
        self.rect.y = paddle_position[1]


class BallSprite(pygame.sprite.Sprite):
    """
    Display ball as pygame sprite
    """

    def __init__(self):
        super().__init__()

        self.image = pygame.Surface([BALL_RADIUS * 2, BALL_RADIUS * 2])
        self.image.fill(COLOR_BACKGROUND)
        self.image.set_colorkey(COLOR_BACKGROUND)

        pygame.draw.circle(
            self.image, COLOR_SPRITE, (BALL_RADIUS, BALL_RADIUS), BALL_RADIUS
        )
        self.rect = self.image.get_rect()

    def update(self, ball_position):
        # Because Sprites center is on top left, adjust sprite position with - BALL_RADIUS
        self.rect.x = ball_position[0] - BALL_RADIUS
        self.rect.y = ball_position[1] - BALL_RADIUS


class Scoreboard:
    def __init__(self):
        # Initialize Font. Takes a few seconds
        self.font = pygame.font.SysFont('FreeMono.ttf', 48)

    def render_score(self, screen, score):
        self.text = self.font.render(
            f'{score[0]} : {score[1]}', True, COLOR_FOREGROUND)
        screen.blit(self.text, (0, 0))


def get_pygame_keys():
    """
    Get keyboard input from pygame for manual playing. Currently NOT IN USE.
    """
    # PyGame event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return ()

    # Paddle Movement. Use parsed arguments for movement
    # if there are any. If none, use pygame keyboard input
    keys = pygame.key.get_pressed()
    if left_movement == "":
        if keys[pygame.K_s]:
            left_paddle.move("down")
        if keys[pygame.K_w]:
            left_paddle.move("up")
    else:
        left_paddle.move(left_movement)

    if right_movement == "":
        if keys[pygame.K_k]:
            right_paddle.move("down")
        if keys[pygame.K_i]:
            right_paddle.move("up")
    else:
        right_paddle.move(right_movement)

    ball.update(left_paddle, right_paddle, score)


class PongGame:
    '''
    Put everything together to make the game Pong.
    '''

    def __init__(self, graphics_enabled=True):
        '''
        Create pygame environment with screen an sprites for paddles and ball.
        If graphics_enabled = False no graphics will be displayed, but game will play the same.
        '''
        # Logical game Objects
        self.left_paddle = Paddle('left')
        self.right_paddle = Paddle('right')
        self.ball = Ball()
        self.score = [0, 0]

        self.graphics_enabled = graphics_enabled
        # Pygame Graphics
        if graphics_enabled:
            pygame.init()
            self.sprite_group = pygame.sprite.Group()
            self.left_paddle_sprite = PaddleSprite()
            self.right_paddle_sprite = PaddleSprite()
            self.ball_sprite = BallSprite()

            self.sprite_group.add(self.left_paddle_sprite,
                                  self.right_paddle_sprite, self.ball_sprite)

            self.scoreboard = Scoreboard()
            self.screen = pygame.display.set_mode(WINDOW_SIZE)

    def reset(self):
        self.ball.reset()

    def tick(self, left_movement, right_movement):
        '''
        Perform one game tick.
        Input: Movement for left and right paddle. ('up', 'stay', 'down')
        Output: terminated = True if a new game started
        '''

        self.left_paddle.move(left_movement)
        self.right_paddle.move(right_movement)
        side_collision, right_paddle_collision = self.ball.update(
            self.left_paddle, self.right_paddle)

        # Calculate new score
        terminated = True
        if side_collision == 'left':
            self.score[1] += 1

        elif side_collision == 'right':
            self.score[0] += 1

        else:
            terminated = False

        if self.graphics_enabled:
            self.update_screen()

        return terminated, right_paddle_collision

    def update_screen(self):
        '''
        Update pygame window with new sprite positions and new score.
        '''
        if not self.graphics_enabled:
            return

        # Update Sprites
        self.left_paddle_sprite.update(self.left_paddle.position)
        self.right_paddle_sprite.update(self.right_paddle.position)
        self.ball_sprite.update(self.ball.position)

        self.screen.fill(COLOR_BACKGROUND)
        pygame.draw.line(self.screen, COLOR_FOREGROUND, [
                         WINDOW_SIZE[0] // 2, 0], [WINDOW_SIZE[0] // 2, WINDOW_SIZE[1]], 5)

        # Draw Sprites
        self.sprite_group.draw(self.screen)

        # Update Scoreboard
        self.scoreboard.render_score(self.screen, self.score)
        pygame.display.flip()
