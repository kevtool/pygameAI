import pygame
import random

class TiledObject:
    def __init__(self, init_row=0, init_col=0, **kwargs):
        required_args = {'tile_rows', 'tile_cols', 'tile_size'}
        for key in required_args:
            if key not in kwargs: raise TypeError('Invalid initialization for TiledObject')

        for key, value in kwargs.items():
            setattr(self, key, value)

        assert 0 <= init_row < self.tile_rows
        assert 0 <= init_col < self.tile_cols
        self.init_row = init_row
        self.init_col = init_col

        self.width = self.tile_size
        self.height = self.tile_size
        self.reset_pos()

    def reset_pos(self):
        self.tile_row = self.init_row
        self.tile_col = self.init_col

        self.set_pos()

    def set_pos(self):
        self.top = 100 + self.tile_row * self.tile_size
        self.left = self.tile_col * self.tile_size

class TiledPlayer(TiledObject, pygame.Rect):
    def __init__(self, **kwargs):
        super(TiledPlayer, self).__init__(**kwargs)

    def move(self, direction):
        match direction:
            case 0:
                # move up
                self.tile_row = max(0, self.tile_row - 1)
            case 1:
                # move left
                self.tile_col = max(0, self.tile_col - 1)
            case 2:
                # move down
                self.tile_row = min(self.tile_rows - 1, self.tile_row + 1)
            case 3:
                # move right
                self.tile_col = min(self.tile_cols - 1, self.tile_col + 1)
            case _:
                raise ValueError('Invalid direction value')

        self.set_pos()

class TiledFood(TiledObject, pygame.Rect):
    def __init__(self, **kwargs):
        super(TiledFood, self).__init__(**kwargs)
    
    def spawn(self, player_row=None, player_col=None):
        row = player_row if player_row is not None else self.tile_row
        col = player_col if player_col is not None else self.tile_col

        self.tile_row = row
        self.tile_col = col

        while self.tile_row == row and self.tile_col == col:
            row = random.randint(0, self.tile_rows - 1)
            col = random.randint(0, self.tile_cols - 1)

        self.tile_row = row
        self.tile_col = col
    
        self.set_pos()