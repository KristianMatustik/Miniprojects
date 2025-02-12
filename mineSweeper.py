import pygame as pg
import numpy as np
import math

class mineField:
    #explore grid
    EXPLORED=1
    UNEXPLORED=0
    FLAGGED=-1

    #mines grid
    MINE=-1
    EMPTY=0
    #MINES IN NEIGHBORHOOD =1,2,3,...

    #game state
    RUNNING = -1
    LOST = 0
    WON = 1

    @staticmethod
    def is_valid(x, y, cols, rows):
        return -1 < x < cols and -1 < y < rows


def newGrid(COLS, ROWS, N_MINES):
    if N_MINES>COLS*ROWS-1:
        N_MINES=COLS*ROWS-1
    grid = np.full((COLS, ROWS), mineField.EMPTY, dtype=int)
    explored = np.full((COLS, ROWS), mineField.UNEXPLORED, dtype=int)

    mine_positions = set(np.random.choice(COLS * ROWS, N_MINES, replace=False))
    for pos in mine_positions:
        x, y = divmod(pos, ROWS)
        grid[x][y] = mineField.MINE

    for x, col in enumerate(grid):
        for y, _ in enumerate(col):
            if grid[x][y]==mineField.MINE:
                continue
            count=0
            for dx in range(-1,2):
                for dy in range(-1,2):
                    nx,ny = x+dx,y+dy
                    if (dx==0 and dy==0) or not mineField.is_valid(nx,ny,COLS,ROWS):
                        continue
                    if grid[nx][ny]==mineField.MINE:
                        count+=1
            grid[x][y]=count
    
    return grid, explored

def explore(grid, explored, x, y):
    toOpen = [(x,y)]
    if explored[x][y]==mineField.EXPLORED:
        return
    explored[x][y]=mineField.EXPLORED
    if grid[x][y]==mineField.MINE:
        for x, col in enumerate(grid):
            for y, _ in enumerate(col):
                if grid[x][y]==mineField.MINE:
                    explored[x][y]=mineField.EXPLORED
        return
    while len(toOpen)>0:
        x,y = toOpen.pop()
        if grid[x][y]==mineField.EMPTY:
            for dx in range(-1,2):
                for dy in range(-1,2):
                    nx,ny = x+dx,y+dy
                    if (dx==0 and dy==0) or not mineField.is_valid(nx,ny,len(grid),len(grid[0])) or explored[nx][ny]==mineField.EXPLORED:
                        continue
                    toOpen.append((nx,ny))
                    explored[nx][ny]=mineField.EXPLORED

def gameOver(state):
    running = True
  
    text1 = font_gameover.render("YOU WON!" if state==1 else "GAME OVER", True, C_TEXT1)
    rect1 = pg.Rect(0, 0.3*HEIGHT, WIDTH, 0.2*HEIGHT)
    text_rect1 = text1.get_rect(center=rect1.center)

    text2 = font_restart.render("press any key to restart", True, C_TEXT2)
    rect2 = pg.Rect(0, 0.4*HEIGHT, WIDTH, 0.2*HEIGHT)
    text_rect2 = text2.get_rect(center=rect2.center)

    screen.blit(text1, text_rect1)
    screen.blit(text2, text_rect2)
    pg.display.flip()
    while(True):
        event = pg.event.wait()   
        if event.type == pg.QUIT:
            running = False
            break            
        if event.type == pg.KEYDOWN:
            break
        if event.type == pg.MOUSEBUTTONUP:
            break
    mines, explored = newGrid(COLS, ROWS, N_MINES)
    return running, mines, explored

def checkWin(mines, explored, N_MINES):
    notExplored = len(explored)*len(explored[0])
    for x, col in enumerate(explored):
        for y, val in enumerate(col):
            if val==mineField.EXPLORED:
                notExplored-=1
                if mines[x][y]==mineField.MINE:
                    return mineField.LOST
    
    return mineField.WON if notExplored==N_MINES else mineField.RUNNING

def solveStep(mines, explored): #ALG for solving, how most people do it. Could be improved only with number of mines for possible endgame...
    flagged=False               #...with some more complex logic, otherwise when no optimal play exists it waits for player input
    for x, col in enumerate(mines):
        for y, _ in enumerate(col):
            count = 0
            if explored[x][y]==mineField.EXPLORED and mines[x][y]!=0:
                for dx in range(-1,2):
                    for dy in range(-1,2):
                        nx,ny = x+dx,y+dy
                        if (dx==0 and dy==0) or not mineField.is_valid(nx,ny,len(mines),len(mines[0])) or explored[nx][ny]==mineField.EXPLORED:
                            continue
                        count+=1
                if mines[x][y]==count:
                    for dx in range(-1,2):
                        for dy in range(-1,2):
                            nx,ny = x+dx,y+dy
                            if (dx==0 and dy==0) or not mineField.is_valid(nx,ny,len(mines),len(mines[0])) or explored[nx][ny]==mineField.EXPLORED or explored[nx][ny]==mineField.FLAGGED:
                                continue
                            explored[nx][ny]=mineField.FLAGGED
                            flagged=True                           
    if not flagged:
        for x, col in enumerate(mines):
            for y, _ in enumerate(col):
                count = 0
                if explored[x][y]==mineField.EXPLORED and mines[x][y]!=0:
                    for dx in range(-1,2):
                        for dy in range(-1,2):
                            nx,ny = x+dx,y+dy
                            if (dx==0 and dy==0) or not mineField.is_valid(nx,ny,len(mines),len(mines[0])) or explored[nx][ny]==mineField.EXPLORED or explored[nx][ny]==mineField.UNEXPLORED:
                                continue
                            count+=1
                    if mines[x][y]==count:
                        for dx in range(-1,2):
                            for dy in range(-1,2):
                                nx,ny = x+dx,y+dy
                                if (dx==0 and dy==0) or not mineField.is_valid(nx,ny,len(mines),len(mines[0])) or explored[nx][ny]==mineField.EXPLORED or explored[nx][ny]==mineField.FLAGGED:
                                    continue
                                explore(mines,explored,nx,ny)




pg.init()

# Constants
COLS, ROWS = 15, 15
N_MINES = 30

WIDTH, HEIGHT = 900, 900
CELL_W, CELL_H = WIDTH//COLS, HEIGHT//ROWS
FONT_SIZE_MINES = math.floor(min(CELL_W,CELL_H)*0.9)
FONT_SIZE_GAMEOVER = math.floor(WIDTH*0.8/9)
FONT_SIZE_RESTART = math.floor(WIDTH*0.8/15)

font_mines = pg.font.Font(None, FONT_SIZE_MINES)
font_gameover = pg.font.Font(None, FONT_SIZE_GAMEOVER)
font_restart = pg.font.Font(None, FONT_SIZE_RESTART)

# Colors
C_GRID = (100, 100, 100)
C_UNEXPLORED = (200, 200, 200)
C_OPEN = (255, 255, 255)
C_FLAG = (255, 0, 0)
C_BOMB = (0, 0, 0)
C_TEXT1 = (0, 0, 255)
C_TEXT2 = (0, 200, 0)

# Game fields
mines, explored = newGrid(COLS, ROWS, N_MINES)
            
# Window setup
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("MineSweeper")

running = True
while running:
    # Event handling
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            event.key == pg.K_SPACE
            solveStep(mines,explored)   #solving algorithm, kinda cheat, would disable in a gmae :) (works only with the info known to player though, doesnt know unrevealed mines)
        elif event.type == pg.MOUSEBUTTONUP:
            x, y = event.pos
            x//=CELL_W
            y//=CELL_H
            button = event.button
            if button == pg.BUTTON_LEFT and explored[x][y]==mineField.UNEXPLORED:
                explore(mines,explored,x,y)
            elif button == pg.BUTTON_RIGHT and explored[x][y]==mineField.UNEXPLORED:
                explored[x][y]=mineField.FLAGGED
            elif button == pg.BUTTON_RIGHT and explored[x][y]==mineField.FLAGGED:
                explored[x][y]=mineField.UNEXPLORED


    # Draw grid
    for row in range(ROWS):
        for col in range(COLS):
            rect = pg.Rect(col*CELL_W, row*CELL_H, CELL_W, CELL_H)
            if explored[col][row] == mineField.UNEXPLORED:
                pg.draw.rect(screen, C_UNEXPLORED, rect)
            elif explored[col][row] == mineField.EXPLORED:
                if mines[col][row]==mineField.EMPTY:
                    pg.draw.rect(screen, C_OPEN, rect)
                elif mines[col][row]==mineField.MINE:
                    pg.draw.rect(screen, C_BOMB, rect)  
                else:
                    pg.draw.rect(screen, C_OPEN, rect)
                    text = font_mines.render(f"{mines[col][row]}", True, C_BOMB)
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)
                                     
            if explored[col][row] == mineField.FLAGGED:
                pg.draw.rect(screen, C_FLAG, rect)
                #pg.draw.circle(screen, C_FLAG,rect.center,min(CELL_H/2,CELL_W/2))

            pg.draw.rect(screen, C_GRID, rect, 1)

    state=checkWin(mines, explored, N_MINES)
    if (state==-1):
        pass
    else:
        running, mines, explored = gameOver(state)

    pg.display.flip()