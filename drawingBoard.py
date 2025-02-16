import pygame as pg
import numpy as np
import math


# Used in testing_NN.py to draw numbers in 28x28 grid for testing

class DrawingBoard:
    def __init__(self, pos, cols, rows, width, height, pen_width=1, pen_color=(255, 255, 255)):
        self.pos = pos
        self.cols = cols
        self.rows = rows
        self.width = width
        self.height = height
        self.pen_width = pen_width
        self.pen_color = pen_color
        self.grid = np.zeros((rows, cols))

    def paint(self, pos):
        pos_x = (pos[0] - self.pos[0]) * self.cols / self.width 
        pos_y = (pos[1] - self.pos[1]) * self.rows / self.height
        for i in range(max(0, math.floor(pos_x - 2*self.pen_width)), min(self.cols, math.ceil(pos_x + 2*self.pen_width))):
            for j in range(max(0, math.floor(pos_y - 2*self.pen_width)), min(self.rows, math.ceil(pos_y + 2*self.pen_width))):
                distance = np.sqrt((i - pos_x + 0.5) ** 2 + (j - pos_y + 0.5) ** 2)
                if distance <= self.pen_width:
                    self.grid[i][j] = 1
                elif distance <= 2 * self.pen_width:
                    self.grid[i][j] = max(2 - distance / self.pen_width, self.grid[i][j])

    def erase(self, pos):
        pos_x = (pos[0] - self.pos[0]) * self.cols / self.width 
        pos_y = (pos[1] - self.pos[1]) * self.rows / self.height
        for i in range(max(0, math.floor(pos_x - 2*self.pen_width)), min(self.cols, math.ceil(pos_x + 2*self.pen_width))):
            for j in range(max(0, math.floor(pos_y - 2*self.pen_width)), min(self.rows, math.ceil(pos_y + 2*self.pen_width))):
                self.grid[i][j] = 0


    def draw(self, screen):
        screen.fill((0, 0, 0))
        for i in range(self.cols):
            for j in range(self.rows):
                color = (self.pen_color[0] * self.grid[i][j], self.pen_color[1] * self.grid[i][j], self.pen_color[2] * self.grid[i][j])
                pg.draw.rect(screen, color, (self.pos[0] + i * self.width // self.cols, self.pos[1] + j * self.height // self.rows, self.width // self.cols, self.height // self.rows))
        for i in range(self.cols + 1):
            pg.draw.line(screen, (50, 50, 50), (self.pos[0] + i * self.width // self.cols, self.pos[1]), (self.pos[0] + i * self.width // self.cols, self.pos[1] + self.height))
        for j in range(self.rows + 1):
            pg.draw.line(screen, (50, 50, 50), (self.pos[0], self.pos[1] + j * self.height // self.rows), (self.pos[0] + self.width, self.pos[1] + j * self.height // self.rows))

    def clear(self):
        self.grid.fill(0)