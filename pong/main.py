# pong game to be played by ML algorithms
#


class Paddle:
    def __init__(side):
        self.side = side if side in ["left", "right"] else "left"    # left = left paddle, right = right paddle
        self.position_y = 0
        self.speed = 0

    def move(direction):
        pass

    
class Ball:
    def __init__():
        # Movement Attributes as 2D-Vectors
        self.position = [0, 0]
        self.speed = [0, 0]


if __name__ == "__main__":
    left_paddle_object = Paddle("left")
    right_paddle_object = Paddle("right")
