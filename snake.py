import pygame as pg
import random

class Snake:
    body = []
    direction = (0,1)
    GRID_WIDTH, GRID_HEIGHT = 20, 20
    
    def __init__(self, W, H, length=3):
        self.GRID_WIDTH = W
        self.GRID_HEIGHT = H
        self.body = [(0, i) for i in range(min(length, H // 2))]
        self.direction = (0, 1)
        

    def move(self, apple):
        head = self.body[-1]
        newHead = ((head[0]+self.direction[0]) %self.GRID_WIDTH, (head[1]+self.direction[1]) %self.GRID_HEIGHT)
        if (newHead==self.body[-2]):
            newHead = ((head[0]-self.direction[0]) %self.GRID_WIDTH, (head[1]-self.direction[1]) %self.GRID_HEIGHT)

        if (apple!=newHead):
            self.body.pop(0)

        for cell in self.body:
            if newHead==cell:
                return False
                                  
        self.body.append(newHead)
        return True


def generate_apple(snake, COLS, ROWS):
    all_cells = {(x, y) for x in range(COLS) for y in range(ROWS)}
    snake_cells = set(snake.body)
    available = list(all_cells - snake_cells)
    return available[random.randint(0, len(available)-1)] if available else None


pg.init()

# Constants
WIDTH, HEIGHT = 800, 800
COLS, ROWS = 20, 20
CELL_W, CELL_H = WIDTH//COLS, HEIGHT//ROWS
FPS = 10

# Colors
C_BACKGROUND = (255, 255, 255)
C_GRID = (200, 200, 200)
C_APPLE = (255, 0, 0)
C_TAIL = (0, 255, 0)
C_HEAD = (0, 0, 255)

# Window setup
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Snake")

# Objects
apple = (COLS//2, ROWS//2)
snake = Snake(COLS,ROWS)

# Main loop
running = True
clock = pg.time.Clock()
while running:
    # Event handling
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_UP:
                snake.direction=(0, -1)
            elif event.key == pg.K_DOWN:
                snake.direction=(0, 1)
            elif event.key == pg.K_LEFT:
                snake.direction=(-1, 0)
            elif event.key == pg.K_RIGHT:
                snake.direction=(1, 0)

    # Move snake, restart if failed
    if(not snake.move(apple)):
        snake=Snake(COLS, ROWS, 3)
        x,y = generate_apple(snake,COLS,ROWS)
        continue

    # Generate new apple if eaten
    if (apple==snake.body[-1]):
        apple = generate_apple(snake,COLS,ROWS)

    # Clear screen
    screen.fill(C_BACKGROUND)  

    # Draw grid
    for row in range(ROWS):
        for col in range(COLS):
            rect = pg.Rect(col*CELL_W, row*CELL_H, CELL_W, CELL_H)
            pg.draw.rect(screen, C_GRID, rect, 1)

    # Draw snake
    for i, cell in enumerate(snake.body):    
        l=len(snake.body)  
        COLOR = (C_TAIL[0]*(l-i)/l + C_HEAD[0]*i/l, C_TAIL[1]*(l-i)/l + C_HEAD[1]*i/l, C_TAIL[2]*(l-i)/l + C_HEAD[2]*i/l)
        pg.draw.rect(screen, COLOR, (cell[0]*CELL_W, cell[1]*CELL_H, CELL_W, CELL_H))

    # Draw apple
    pg.draw.rect(screen, C_APPLE, (apple[0]*CELL_W, apple[1]*CELL_H, CELL_W, CELL_H))

    # Update display
    pg.display.flip() 

    # Game speed
    clock.tick(FPS)

pg.quit()