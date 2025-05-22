import numpy as np
from .snake import SnakeGame, Direction, Point

class SnakeEnv:
    def __init__(self):
        self.game = SnakeGame()

    def reset(self):
        self.game.reset()
        return self.get_state()

    def step(self, action):
        reward, done, score = self.game.play_step(action)
        state = self.get_state()
        return state, reward, done, score

    def get_state(self):
        head = self.game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = self.game.direction == Direction.LEFT
        dir_r = self.game.direction == Direction.RIGHT
        dir_u = self.game.direction == Direction.UP
        dir_d = self.game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.game._is_collision(point_r)) or
            (dir_l and self.game._is_collision(point_l)) or
            (dir_u and self.game._is_collision(point_u)) or
            (dir_d and self.game._is_collision(point_d)),

            # Danger right
            (dir_u and self.game._is_collision(point_r)) or
            (dir_d and self.game._is_collision(point_l)) or
            (dir_l and self.game._is_collision(point_u)) or
            (dir_r and self.game._is_collision(point_d)),

            # Danger left
            (dir_d and self.game._is_collision(point_r)) or
            (dir_u and self.game._is_collision(point_l)) or
            (dir_r and self.game._is_collision(point_u)) or
            (dir_l and self.game._is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.game.food.x < self.game.head.x,  # food left
            self.game.food.x > self.game.head.x,  # food right
            self.game.food.y < self.game.head.y,  # food up
            self.game.food.y > self.game.head.y   # food down
        ]

        return np.array(state, dtype=int)
