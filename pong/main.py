# Pong game to be played by ML algorithms
#

import numpy as np
import random
import math
import time
import pygame
from typing import List, Tuple

WINDOW_SIZE = (800, 500) # (0|0) is on the bottom left
BALL_RADIUS = 12
BALL_SPEED = 1
PADDLE_SPEED = 1
PADDLE_WIDTH = 20
PADDLE_HEIGHT = 100
COLOR_WINDOW_BACKGROUND = (0, 0, 0)
COLOR_SPRITE_MAIN = (255, 255, 255)
FPS_LIMIT = 60


class Paddle:
    def __init__(self, side):
        self.side = side if side in ['left', 'right'] else 'left'    # left = left paddle, right = right paddle
        self.position_y = WINDOW_SIZE[1] / 2    # y-position of lowermost part of paddle
        

    def move(self,_direction:_str) -> None:
        # Handle movement and top / bottom Collision
        if direction == 'stay':
            pass

        elif direction == 'up':
            self.position_y = min(self.position_y + PADDLE_SPEED, WINDOW_SIZE[1] - PADDLE_HEIGHT)

        elif direction == 'down':
            self.position_y = max(self.position_y - PADDLE_SPEED, 0)

            
class Ball:
    def __init__(self):
        self.position = np.array([WINDOW_SIZE[0] / 2, WINDOW_SIZE[1] / 2])
        
        # Ball should start at a random angle. Achieved by setting random starting y-Velocity
        # Multiply by 0.9 that ball doesn't start with only vertical movement
        random_y_velocity = random.uniform(-0.9 * BALL_SPEED, 0.9 * BALL_SPEED)

        # norm of velocity must = BALL_SPEED
        self.velocity = np.array([math.sqrt(BALL_SPEED - self.random_y_velocity ** 2), random_y_velocity])

    def wall_collision(self) -> None:
        if True:
            # TODO: Add checks for collision
            self.velocity.dot(np.array)

    def paddle_collision(self) -> None:
        if True:
            # TODO: Add checks for collision
            np.multiply(self.velocity[0], np.array([-1, 1]))

    def update(self) -> None:
        self.wall_collision()
        self.paddle_collision()

        self.position += self.velocity
        
        return self.position, self.velocity


class PaddleSprite(pygame.sprite.Sprite):
    ''' 
    Display paddles as pygame sprites
    '''
    
    def __init__(self):
        super().__init__()
        
        self.image = pygame.Surface([PADDLE_WIDTH, PADDLE_HEIGHT])
        self.image.fill(COLOR_WINDOW_BACKGROUND)
        self.image.set_colorkey(COLOR_WINDOW_BACKGROUND)

        pygame.draw.rect(self.image, COLOR_SPRITE_MAIN, [0, 0, PADDLE_WIDTH, PADDLE_HEIGHT])
        self.rect = self.image.get_rect()


class BallSprite(pygame.sprite.Sprite):
    '''
    Display ball as pygame sprite
    '''

    def __init__(self):
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.image.fill(COLOR_WINDOW_BACKGROUND)
        self.image.set_colorkey(COLOR_WINDOW_BACKGROUND)

        pygame.draw.circle(self.image, COLOR_SPRITE_MAIN, (0, 0), BALL_RADIUS) 
    
def tick(left_paddle, right_paddle, ball, left_movement='', right_movement='') -> Tuple:
    '''
    Handle one game Tick. Uses parsed arguments for movement if given, else pygame keyboard input. 
    Movement options are 'stay', 'up', 'down'.
    '''
    # PyGame event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return ()
        
    # Use parsed arguments for movement if there are any. If none, use pygame keyboard input
    keys = pygame.key.get_pressed()
    if left_movement == '':
        if keys[pygame.K_s]:
            left_paddle.move('down')
        if keys[pygame.K_w]:
            left_paddle.move('up')
    else:
        left_paddle.move(left_movement)
                
    if right_movement == '':
        if keys[pygame.K_k]:
            right_paddle.move('down')
        if keys[pygame.K_i]:
            right_paddle.move('up')
    else:
        right_paddle.move(right_movement)

    ball.update()

    return left_paddle.position_y, right_paddle.position_y, ball.position, ball.velocity

                        
def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption('Pong')
    
    left_paddle = Paddle('left')
    right_paddle = Paddle('right')
    ball = Ball()

    # Game loop
    clock = pygame.time.Clock()
    while tick(left_paddle, right_paddle, ball):
        clock.tick(FPS_LIMIT)
        continue

                        
if __name__ == '__main__':
    main()
        
