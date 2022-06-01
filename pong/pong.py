# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of maturaarbeit_code.
# maturaarbeit_code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# maturaarbeit_code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with maturaarbeit_code. If not, see <https://www.gnu.org/licenses/>.
# Pong game to be played by ML algorithms
import math
import random
import time
from typing import List
from typing import Tuple

import numpy as np
import pygame

# Adjust as needed
# (0|0) is on the top left. x-Axis is right, y-Axis down
WINDOW_SIZE = (800, 500)
BALL_RADIUS = 12
BALL_SPEED = 4
PADDLE_SPEED = 3
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

    def move(self, direction: str) -> None:
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

    def relative_y_position(self) -> float:
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

    def wall_collision(self, score) -> None:
        if (self.position[0] - BALL_RADIUS) <= 0:
            # Collision with left wall
            score[1] += 1
            self.reset()

        if (self.position[0] + BALL_RADIUS) >= WINDOW_SIZE[0]:
            # Collision with right wall
            score[0] += 1
            self.reset()

        if (self.position[1] - BALL_RADIUS) <= 0 or (
            self.position[1] + BALL_RADIUS
        ) >= WINDOW_SIZE[1]:
            # Collision with top or bottom wall
            self.velocity = np.multiply(self.velocity, np.array([1, -1]))

    def paddle_collision(self, left_paddle, right_paddle) -> None:
        # Apply rectangle-circle collision function to ball with both paddles
        self.velocity = rect_circle_collision(
            (
                left_paddle.position[0],
                left_paddle.position[1],
                left_paddle.position[0] + PADDLE_WIDTH,
                left_paddle.position[1] + PADDLE_HEIGHT,
            ),
            (self.position[0], self.position[1], BALL_RADIUS),
            self.velocity,
        )
        self.velocity = rect_circle_collision(
            (
                right_paddle.position[0],
                right_paddle.position[1],
                right_paddle.position[0] + PADDLE_WIDTH,
                right_paddle.position[1] + PADDLE_HEIGHT,
            ),
            (self.position[0], self.position[1], BALL_RADIUS),
            self.velocity,
        )

    def update(self, left_paddle, right_paddle, score) -> None:
        self.wall_collision(score)
        self.paddle_collision(left_paddle, right_paddle)

        self.position += self.velocity

    def relative_position(self) -> Tuple[float, float]:
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


class PongGraphics:
    '''
    Handles all graphics/pygame related stuff for pong game.
    '''

    def __init__(self):
        '''
        Create pygame environment with screen an sprites for paddles and ball.
        '''
        pygame.init()
        self.sprite_group = pygame.sprite.Group()
        self.left_paddle_sprite = PaddleSprite()
        self.right_paddle_sprite = PaddleSprite()
        self.ball_sprite = BallSprite()

        self.sprite_group.add(self.left_paddle_sprite,
                              self.right_paddle_sprite, self.ball_sprite)

        self.scoreboard = Scoreboard()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)

    def update_screen(self, left_paddle_position, right_paddle_position, ball_position, score):
        '''
        Update Screen with new sprite positions and new score.
        '''
        # Update Sprites
        self.left_paddle_sprite.update(left_paddle_position)
        self.right_paddle_sprite.update(right_paddle_position)
        self.ball_sprite.update(ball_position)

        self.screen.fill(COLOR_BACKGROUND)
        pygame.draw.line(self.screen, COLOR_FOREGROUND, [
                         WINDOW_SIZE[0] // 2, 0], [WINDOW_SIZE[0] // 2, WINDOW_SIZE[1]], 5)

        # Draw Sprites
        self.sprite_group.draw(self.screen)

        # Update Scoreboard
        self.scoreboard.render_score(self.screen, score)
        pygame.display.flip()


def tick(left_paddle, right_paddle, ball, score, left_movement, right_movement, pygame_graphics):
    """
    Handle one game Tick. Uses parsed arguments for movement if given, else pygame keyboard input.
    Movement options are 'stay', 'up', 'down'.
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

    # Perform graphics operations
    pygame_graphics.update_screen(
        left_paddle.position, right_paddle.position, ball.position, score)

    return (
        left_paddle.relative_y_position(),
        right_paddle.relative_y_position(),
        ball.relative_position(),
        ball.velocity,
    )


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
            return np.multiply(circle_velocity, np.array([-1, 1]))

    # Bottom side
    if circle[1] > rectangle[1] and rectangle[0] < circle[0] < rectangle[2]:
        # Collision?
        if circle[1] - circle[2] <= rectangle[1]:
            return np.multiply(circle_velocity, np.array([1, -1]))

    # Right side
    if circle[0] > rectangle[2] and rectangle[1] < circle[1] < rectangle[3]:
        # Collision?
        if circle[0] - circle[2] <= rectangle[2]:
            return np.multiply(circle_velocity, np.array([-1, 1]))

    # Top side
    if circle[1] < rectangle[1] and rectangle[0] < circle[1] < rectangle[2]:
        # Collision?
        if circle[1] + circle[2] >= rectangle[1]:
            return np.multiply(circle_velocity, np.array([1, -1]))

    # If the ball is not facing any sides, check if it makes contact with any corner
    # Top left corner
    if (rectangle[0] - circle[0]) ** 2 + (rectangle[1] - circle[1]) ** 2 <= circle[
        2
    ] ** 2:
        return circle_corner_bounce(
            (rectangle[0], rectangle[1]), circle, circle_velocity
        )

    # Bottom left corner
    if (rectangle[0] - circle[0]) ** 2 + (rectangle[3] - circle[1]) ** 2 <= circle[
        2
    ] ** 2:
        return circle_corner_bounce(
            (rectangle[0], rectangle[3]), circle, circle_velocity
        )

    # Bottom right corner
    if (rectangle[2] - circle[0]) ** 2 + (rectangle[3] - circle[1]) ** 2 <= circle[
        2
    ] ** 2:
        return circle_corner_bounce(
            (rectangle[2], rectangle[3]), circle, circle_velocity
        )

    # Top right corner
    if (rectangle[2] - circle[0]) ** 2 + (rectangle[1] - circle[1]) ** 2 <= circle[
        2
    ] ** 2:
        return circle_corner_bounce(
            (rectangle[2], rectangle[1]), circle, circle_velocity
        )

    # If none of the above options are applicable, the circle didn't collide, so just return
    # the input velocity
    return circle_velocity


class PongEnv:
    def __init__(self):
        '''
        Reset the game to initial state and return initial state
        '''
        self.left_paddle = Paddle('left')
        self.right_paddle = Paddle('right')
        self.ball = Ball()
        self.score = [0, 0]
        self.pygame_graphics = PongGraphics()
        self.possible_actions = ['up', 'down', 'stay']
        self.state_size = len(self.make_observation())

    def make_observation(self):
        '''
        Helper method returning current game's current internal state.
        '''
        return (self.left_paddle.relative_y_position(), self.right_paddle.relative_y_position(), *self.ball.relative_position(), *self.ball.velocity)

    def step(self, action):
        '''
        Do one game move with given action and return image, reward and wheter or not the game terminates
        '''
        prev_score = self.score.copy()

        # Get desired paddle movement from first (and only) entry of action tuple
        right_movement = action[0]

        # For the time being, make the opponent unmoving
        left_movement = 'stay'
        tick(self.left_paddle,
             self.right_paddle,
             self.ball,
             self.score,
             left_movement,
             right_movement,
             self.pygame_graphics
             )

        if self.score == prev_score:
            # Slight negative reward for each game tick which is not a point
            reward = 0
        elif self.score[0] == prev_score[0] + 1:
            # Negative reward if opponent gets a point
            reward = -1
        elif self.score[1] == prev_score[1] + 1:
            # Positive reward if agent gets a point
            reward = 1

        return self.make_observation(), reward, self.score


def main():
    env = PongEnv()
    clock = pygame.time.Clock()
    while True:
        # print("Step: ", env.step(['stay']), "Score: ", env.score)
        clock.tick(FPS_LIMIT)


def _main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Pong")

    left_paddle = Paddle("left")
    right_paddle = Paddle("right")
    ball = Ball()
    score = [0, 0]

    # Sprites do no math and have no mechanics other than displaying the Ball and Paddles
    sprite_group = pygame.sprite.Group()
    sprites = [PaddleSprite(), PaddleSprite(), BallSprite()]
    for sprite in sprites:
        sprite_group.add(sprite)

    screen = pygame.display.set_mode(WINDOW_SIZE)
    scoreboard = Scoreboard()

    # Game loop
    clock = pygame.time.Clock()
    while True:
        tick(
            left_paddle,
            right_paddle,
            ball,
            screen,
            sprites,
            sprite_group,
            score,
            scoreboard,
        )
        print(score)
        clock.tick(FPS_LIMIT)
        continue


if __name__ == "__main__":
    main()
