# Pong game to be played by ML algorithms
#

import numpy as np
import random
import math
import time
import pygame
from typing import List, Tuple

WINDOW_SIZE = (800, 500) # (0|0) is on the top left. x-Axis is right, y-Axis down
BALL_RADIUS = 12
BALL_SPEED = 2
PADDLE_SPEED = 2
PADDLE_WIDTH = 20
PADDLE_HEIGHT = 100
COLOR_BACKGROUND = (0, 0, 0)
COLOR_FOREGROUND = (100, 100, 100)
COLOR_SPRITE = (255, 255, 255)
FPS_LIMIT = 60
GRAPHICAL_MODE = True


class Scoreboard:
    def __init__(self):
        self.font = pygame.font.SysFont('FreeMono.ttf', 48) # Initialize Font. Takes a few seconds
        
    def render_score(self, screen, score):
        self.text = self.font.render(f'{score[0]} : {score[1]}', True, COLOR_FOREGROUND)
        screen.blit(self.text, (0, 0))
        
        
class Paddle:
    def __init__(self, side):
        self.side = side if side in ['left', 'right'] else 'left'    # left = left paddle, right = right paddle
        self.position = [0, (WINDOW_SIZE[1] - PADDLE_HEIGHT) / 2]    # Start paddle in the middle (vertically) of according side
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

    def relative_y_position(self) -> float:
        ''' 
        Returns a value from 0 to 1 indicating how far the paddle has travelled
        '''

        # Divide the absolute y-position by the window height - paddle height.
        # Because the paddle can only move as far down as it's height allows
        # The subraction is that resizing the paddle doesn't change the values
        return self.position[1] / (WINDOW_SIZE[1] - PADDLE_HEIGHT)

            
class Ball:
    def __init__(self):
        self.position = np.array([WINDOW_SIZE[0] / 2, WINDOW_SIZE[1] / 2])

        # self.position = np.array([WINDOW_SIZE[0] - BALL_RADIUS, WINDOW_SIZE[1] - BALL_RADIUS]) # For debugging
        # Ball should start at a random angle. Achieved by setting random starting y-Velocity
        # Multiply by 0.9 that ball doesn't start with only vertical movement
        random_y_velocity = random.uniform(-0.9 * BALL_SPEED, 0.9 * BALL_SPEED)

        # norm of velocity must = BALL_SPEED
        self.velocity = np.array([math.sqrt(BALL_SPEED ** 2- random_y_velocity ** 2), random_y_velocity])

    def reset(self):
        self.__init__()
        
    def wall_collision(self, score) -> None:
        # TODO: Implement scoring and reset function
        if (self.position[0] - BALL_RADIUS) <= 0:
            # Collision with left wall
            score[1] += 1
            self.reset()
            
        if (self.position[0] + BALL_RADIUS) >= WINDOW_SIZE[0]:
            # Collision with right wall
            score[0] += 1
            self.reset()

        if (self.position[1] - BALL_RADIUS) <= 0 or (self.position[1] + BALL_RADIUS) >= WINDOW_SIZE[1]:
            # Collision with top or bottom wall
            self.velocity = np.multiply(self.velocity, np.array([1, -1]))
            

    def paddle_collision(self, left_paddle, right_paddle) -> None:
        left_paddle_collision = (left_paddle.position[1] <= self.position[1] <= left_paddle.position[1] + PADDLE_HEIGHT) and (self.position[0] - BALL_RADIUS <= PADDLE_WIDTH)
        right_paddle_collision =  (right_paddle.position[1] <= self.position[1] <= left_paddle.position[1] + PADDLE_HEIGHT) and (self.position[0] + BALL_RADIUS + PADDLE_WIDTH >= WINDOW_SIZE[0])
        if left_paddle_collision or right_paddle_collision:
            self.velocity = np.multiply(self.velocity, np.array([-1, 1]))

    def update(self, left_paddle, right_paddle, score) -> None:
        self.wall_collision(score)
        self.paddle_collision(left_paddle, right_paddle)

        self.position += self.velocity

    def relative_position(self) -> Tuple[float, float]:
        '''
        Return the balls position in the playing field as two floats from 0 to 1 
        (relative horizontal and vertical position)
        '''

        # Window acessible to ball = window size - 2 * ball radius,
        # as on each side the radius limits how close ball can approach side, so divide by window size - 2 * radius
        #
        # And ball has travelled 0 % when it is "ball radius" away from border, so subtract this from ball position
        return (self.position[0] - BALL_RADIUS) / (WINDOW_SIZE[0] - 2 * BALL_RADIUS), (self.position[1] - BALL_RADIUS) / (WINDOW_SIZE[1] - 2 * BALL_RADIUS)

        
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
        
    def update(self, paddle_position):
        self.rect.x = paddle_position[0]
        self.rect.y = paddle_position[1]
        
        

class BallSprite(pygame.sprite.Sprite):
    '''
    Display ball as pygame sprite
    '''

    def __init__(self):
        super().__init__()

        self.image = pygame.Surface([BALL_RADIUS * 2, BALL_RADIUS * 2])
        self.image.fill(COLOR_BACKGROUND)
        self.image.set_colorkey(COLOR_BACKGROUND)

        pygame.draw.circle(self.image, COLOR_SPRITE, (BALL_RADIUS, BALL_RADIUS), BALL_RADIUS)
        self.rect = self.image.get_rect()

    def update(self, ball_position):
        # Because Sprites center is on top left, adjust sprite position with - BALL_RADIUS
        self.rect.x = ball_position[0] - BALL_RADIUS
        self.rect.y = ball_position[1] - BALL_RADIUS
        

def tick(left_paddle, right_paddle, ball, screen, sprites, sprite_group, score, scoreboard, 
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
    ball.update(left_paddle, right_paddle, score)

    # If graphical mode enabled, perform graphics operations
    if GRAPHICAL_MODE:
        sprites[0].update(left_paddle.position)
        sprites[1].update(right_paddle.position)
        sprites[2].update(ball.position)

        # Draw Sprites
        screen.fill(COLOR_BACKGROUND)
        pygame.draw.line(screen, COLOR_FOREGROUND, [WINDOW_SIZE[0] // 2, 0], [WINDOW_SIZE[0] // 2, WINDOW_SIZE[1]], 5)
        
        sprite_group.draw(screen)
        
        # Update Scoreboard
        scoreboard.render_score(screen, score)
        pygame.display.flip()

    return left_paddle.relative_y_position(), right_paddle.relative_y_position(), ball.relative_position(), ball.velocity


def circle_corner_bounce(corner: Tuple[int, int], circle: Tuple[int, int, int], circle_velocity):
    '''
    Handle collision of a circle with a single point
    '''

    #
    return


def rect_circle_collision(rectangle: Tuple[int, int, int, int], circle: Tuple[int, int, int], circle_velocity):
    '''
    Collide a moving circle with a rectangle. Parse rectangle as top left and bottom right point, circle as centre and radius.
    '''

    # There are 8 possibilites where the ball can be relative to the rectangle.
    # Four are facing each side of the rectangle
    # Four are closest to each corner

    # Left side
    if circle[0] < rectangle[0] and rectangle[1] < circle[1] < rectangle[3]:
        # Collision?
        if circle[0] + circle[2] >= rectangle[0]:
            return np.multiply(circle_velocity, np.array([-1, 1]))

    # Bottom side
    elif circle[1] > rectangle[1] and rectangle[0] < circle[0] < rectangle[2]:
        # Collision?
        if circle[1] - circle[2] <= rectangle[1]:
            return np.multiply(circle_velocity, np.array([1, -1]))

    # Right side
    elif circle[0] > rectangle[2] and rectangle[1] < circle[1] < rectangle[3]:
        # Collision?
        if circle[0] - circle[2] <= rectangle[2]:
            return np.multiply(circle_velocity, np.array([-1, 1]))

    # Top side
    elif circle[1] < rectangle[1] and rectangle[0] < circle[1] < rectangle[2]:
        # Collision?
        if circle[1] + circle[2] >= rectangle[1]:
            return np.multiply(circle_velocity, np.array([1, -1]))

    else:
        # If the ball is not facing any sides, check if it makes contact with any corner
        # Top left corner
        if (rectangle[0] - ball[0]) ** 2 + (rectangle[1] - ball[1]) ** 2 <= circle[2] ** 2:
            pass

        # Bottom left corner
        if (rectangle[0] - ball[0]) ** 2 + (rectangle[3] - ball[1]) ** 2 <= circle[2] ** 2:
            pass

        # Bottom right corner
        if (rectangle[2] - ball[0]) ** 2 + (rectangle[2] - ball[1]) ** 2 <= circle[2] ** 2:
            pass
        
        # Top right corner
        if (rectangle[2] - ball[0]) ** 2 + (rectangle[1] - ball[1]) ** 2 <= circle[2] ** 2:
            pass

def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption('Pong')
    
    left_paddle = Paddle('left')
    right_paddle = Paddle('right')
    ball = Ball()
    score = [0, 0]
    
    # Only define and update sprites if graphical mode is enabled.
    # Sprites do no math and have no mechanics other than displaying the Ball and Paddles
    if GRAPHICAL_MODE:
        sprite_group = pygame.sprite.Group()
        sprites = [PaddleSprite(), PaddleSprite(), BallSprite()]
        for sprite in sprites:
            sprite_group.add(sprite)
            
        screen = pygame.display.set_mode(WINDOW_SIZE)
        scoreboard = Scoreboard()
        
    # Game loop
    clock = pygame.time.Clock()
    while True:
        tick(left_paddle, right_paddle, ball, screen, sprites, sprite_group, score, scoreboard)
        print(score)
        clock.tick(FPS_LIMIT)
        continue

                        
if __name__ == '__main__':
    main()
    
 
