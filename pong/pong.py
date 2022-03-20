# Pong game to be played by ML algorithms
#

import numpy as np
import random
import math
import time
import pygame
from typing import List, Tuple

WINDOW_SIZE = (800, 500) # (0|0) is on the top left. 
BALL_RADIUS = 12
BALL_SPEED = 10
PADDLE_SPEED = 10
PADDLE_WIDTH = 20
PADDLE_HEIGHT = 100
COLOR_BACKGROUND = (0, 0, 0)
COLOR_FOREGROUND = (100, 100, 100)
COLOR_SPRITE = (255, 255, 255)
FPS_LIMIT = 60
GRAPHICAL_MODE = True


class Paddle:
    def __init__(self, side):
        self.side = side if side in ['left', 'right'] else 'left'    # left = left paddle, right = right paddle
        self.position = [0, (WINDOW_SIZE[1] - PADDLE_HEIGHT) / 2]    # Start paddle in the middle of according side
        if self.side == 'left':
            self.position[0] = 0
        elif self.side == 'right':
            self.position[0] = WINDOW_SIZE[0] - PADDLE_WIDTH

    def move(self, direction: str) -> None:
        # Handle movement and top / bottom Collision
        if direction == 'stay':
            pass

        elif direction == 'up':
            self.position[1] = max(self.position[1] - PADDLE_SPEED, 0)

        elif direction == 'down':
            self.position[1] = min(self.position[1] + PADDLE_SPEED, WINDOW_SIZE[1] - PADDLE_HEIGHT)

            
class Ball:
    def __init__(self):
        self.position = np.array([WINDOW_SIZE[0] / 2, WINDOW_SIZE[1] / 2])
        
        # Ball should start at a random angle. Achieved by setting random starting y-Velocity
        # Multiply by 0.9 that ball doesn't start with only vertical movement
        random_y_velocity = random.uniform(-0.9 * BALL_SPEED, 0.9 * BALL_SPEED)

        # norm of velocity must = BALL_SPEED
        self.velocity = np.array([math.sqrt(BALL_SPEED ** 2- random_y_velocity ** 2), random_y_velocity])

    def wall_collision(self) -> None:
        if True:
            # TODO: Add checks for collision
            pass

    def paddle_collision(self) -> None:
        if True:
            # TODO: Add checks for collision
            pass
            np.multiply(self.velocity[0], np.array([-1, 1]))

    def update(self) -> None:
        self.wall_collision()
        self.paddle_collision()

        self.position += self.velocity

        
class PaddleSprite(pygame.sprite.Sprite):
    ''' 
    Display paddles as pygame sprites
    '''
    
    def __init__(self):
        super().__init__()

        self.image = pygame.Surface([PADDLE_WIDTH, PADDLE_HEIGHT])
        self.image.fill(COLOR_BACKGROUND)
        self.image.set_colorkey(COLOR_BACKGROUND)

        pygame.draw.rect(self.image, COLOR_SPRITE, [0, 0, PADDLE_WIDTH, PADDLE_HEIGHT])


        self.rect = self.image.get_rect()
        print(self.rect)
        
    def update(self, paddle_position):
        self.rect.x = paddle_position[0]
        self.rect.y = paddle_position[1]
        
        

class BallSprite(pygame.sprite.Sprite):
    '''
    Display ball as pygame sprite
    '''

    def __init__(self):
        super().__init__()

        self.image = pygame.Surface([2 * BALL_RADIUS, 2 * BALL_RADIUS])
        self.image.fill(COLOR_BACKGROUND)
        self.image.set_colorkey(COLOR_BACKGROUND)

        pygame.draw.circle(self.image, COLOR_SPRITE, (BALL_RADIUS, BALL_RADIUS), BALL_RADIUS) # TODO: If bugs appear, make sure to set the correct ball center
        self.rect = self.image.get_rect()
        
    def update(self, ball_position):
        self.rect.x = ball_position[0]
        self.rect.y = ball_position[1]
        

def pixel_to_relative(coordinates: Tuple[float, float]):
    '''
    Function takes in 2D Vector for Position or Velocity and returns same Vector but adjusted 
    so its components are numbers from 0 to 1
    '''

    # TODO: ACCOMODATE FOR SPRITE SIZE
    return (coordinates[0] / WINDOW_SIZE[0], coordinates[1] / WINDOW_SIZE[1])


def tick(left_paddle, right_paddle, ball, screen, sprites, sprite_group,
         left_movement='', right_movement='') -> Tuple:
    '''
    Handle one game Tick. Uses parsed arguments for movement if given, else pygame keyboard input. 
    Movement options are 'stay', 'up', 'down'.
    '''
    # PyGame event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return ()
        
    # Paddle Movement. Use parsed arguments for movement
    # if there are any. If none, use pygame keyboard input
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

    # print(left_paddle.position, right_paddle.position)
    ball.update()

    # If graphical mode enabled, perform graphics operations
    if GRAPHICAL_MODE:
        # TODO: fix sprite updating not working because sprites parsed wrongly at the moment
        sprites[0].update(left_paddle.position)
        sprites[1].update(right_paddle.position)
        sprites[2].update(ball.position)

        # Draw Sprites
        screen.fill(COLOR_BACKGROUND)
        pygame.draw.line(screen, COLOR_FOREGROUND, [0, 0], [0, 0], 5)

        sprite_group.draw(screen)
        pygame.display.flip()

    return tuple(map(lambda x: pixel_to_relative(x), (left_paddle.position, right_paddle.position, ball.position, ball.velocity))) 

                        
def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption('Pong')
    
    left_paddle = Paddle('left')
    right_paddle = Paddle('right')
    ball = Ball()

    # Only define and update sprites if graphical mode is enabled. Sprites do no math and have no mechanics
    # other than displaying the Ball and Paddles
    if GRAPHICAL_MODE:
        sprite_group = pygame.sprite.Group()
        sprites = [PaddleSprite(), PaddleSprite(), BallSprite()]
        for sprite in sprites:
            sprite_group.add(sprite)
        
        screen = pygame.display.set_mode(WINDOW_SIZE)
        
    # Game loop
    clock = pygame.time.Clock()
    while True:
        print(tick(left_paddle, right_paddle, ball, screen, sprites, sprite_group))
        clock.tick(FPS_LIMIT)
        continue

                        
if __name__ == '__main__':
    main()
        
