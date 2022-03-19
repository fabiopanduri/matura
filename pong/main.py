# pong game to be played by ML algorithms
#
import random
import math
import time
import pygame

WINDOW_SIZE = (800, 500)
COLOR_WINDOW_BACKGROUND = (0, 0, 0)
COLOR_SPRITE_MAIN = (255, 255, 255)
SLEEP_TIME = 0.1    # Time to wait after each tick (in seconds). Adjust to change game speed


class Paddle:
    def __init__(self, side):
        self.side = side if side in ["left", "right"] else "left"    # left = left paddle, right = right paddle
        self.position_y = 0
        self.velocity = 0

    def move(self, direction):
        pass

    
class Ball:
    def __init__(self):
        # Movement Attributes as 2D-Vectors
        self.position = [0, 0]
        self.random_y_velocity = random.uniform(-0.9, 0.9)    # Ball should start at a random angle
        self.velocity = [math.sqrt(1 - self.random_y_velocity ** 2), self.random_y_velocity]    # norm of velocity must = 1

    def wall_collision(self):
        # Check for wall collision and in case of one update velocity
        if True:
            # Add checks for collision
            self.velocity[0] *= -1

    def paddle_collision(self):
        # Check for paddle collision and in case of one update velocity
        if True:
            # Add checks for collision
            self.velocity[0] *= -1

    def update(self):
        self.wall_collision()
        self.paddle_collision()

        # TODO: figure out beautiful vector addition
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]
        
        return self.position, self.velocity

    
def tick(left_paddle, right_paddle, ball):
    # PyGame event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        
    player_left_movement = 0
    player_right_movement = 0
         
    
    left_paddle.move(player_left_movement)
    right_paddle.move(player_right_movement)
    print(ball.update())
    return True


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Pong")
    
    left_paddle = Paddle("left")
    right_paddle = Paddle("right")
    ball = Ball()

    # Game loop 
    while tick(left_paddle, right_paddle, ball):
        time.sleep(SLEEP_TIME)
        continue

        
