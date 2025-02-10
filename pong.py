import math
import pygame as pg
from abc import ABC, abstractmethod
import random

class GameObject(ABC):
    def __init__(self,x,y,w,h,vx=0,vy=0):
        self.pos_x = x
        self.pos_y = y
        self.size_w = w
        self.size_h = h
        self.speed_x = vx
        self.speed_y = vy

    def update(self, dt):
        self.pos_x+=self.speed_x*dt
        self.pos_y+=self.speed_y*dt
  

    #Check if overlaps boundry/object
    def topOverlap(self, y):
        return self.pos_y < y
    def botOverlap(self, y):
        return self.pos_y + self.size_h > y  
    def leftOverlap(self, x):
        return self.pos_x < x   
    def rightOverlap(self, x):
        return self.pos_x + self.size_w > x   
    def vertOverlap(self, other):
        return self.topOverlap(other.pos_y+other.size_h) and self.botOverlap(other.pos_y) 
    def horOverlap(self, other):
        return self.leftOverlap(other.pos_x+other.size_w) and self.rightOverlap(other.pos_x) 
    def totalOverlap(self, other):
        return self.vertOverlap(other) and self.horOverlap(other)
    
    #Correct position and speed if overlaps
    def topBounds(self, y):
        if self.topOverlap(y):
            self.pos_y=y
            self.speed_y*=-1
    def botBounds(self, y):
        if self.botOverlap(y):
            self.pos_y=y-self.size_h
            self.speed_y*=-1
    def leftBounds(self, x):
        if self.leftOverlap(x):
            self.pos_x=x
            self.speed_x*=-1
    def rightBounds(self, x):
        if self.rightOverlap(x):
            self.pos_x=x-self.size_w
            self.speed_x*=-1
    def vertBounds(self, y1, y2):
        self.topBounds(y1)
        self.botBounds(y2)
    def horBounds(self, x1, x2):
        self.leftBounds(x1)
        self.rightBounds(x2)
    def totalBounds(self, y1, y2, x1, x2):
        self.vertBounds(y1,y2)
        self.horBounds(x1,x2)

    @abstractmethod
    def draw(self):
        pass


class PowerUp(GameObject, ABC):
    def __init__(self, x, y, w, h, v, d):
        i = random.choice([0,1,2])
        self.SPEED_P=0
        self.SPEED_B=0
        self.SIZE=0
        if i==0:
            self.SPEED_P=100
            self.COLOR = (0,255,0)
        elif i==1:
            self.SPEED_B=100
            self.COLOR = (255,0,0)
        elif i==2:
            self.SIZE=100
            self.COLOR = (0,0,255)
        self.duration=d
        vx=random.uniform(0.5,1)*v*random.choice([-1,1])
        vy=random.uniform(0.5,1)*v*random.choice([-1,1])
        super().__init__(x, y, w, h, vx, vy)

    def draw(self):
        rect = pg.Rect(self.pos_x, self.pos_y, self.size_w, self.size_h)
        pg.draw.rect(screen, self.COLOR, rect)

    def update(self, dt, H, W):
        super().update(dt)
        self.totalBounds(0,H,0,W)

    def interactPaddle(self, paddle):
        if self.totalOverlap(paddle):
            paddle.activePowerUps.append(self)
            paddle.MAX_V_PADDLE+=self.SPEED_P
            paddle.MAX_V_BALL+=self.SPEED_B
            paddle.size_h+=self.SIZE
            paddle.pos_y-=self.SIZE/2
            paddle.updateColor()
            return True
        return False



class Paddle(GameObject):
    def __init__(self, x, y, w, h, v_P, v_B):
        self.COLOR = (200,200,200)
        self.MAX_V_PADDLE=v_P
        self.MAX_V_BALL=v_B
        self.score=0
        self.activePowerUps=[]
        super().__init__(x, y, w, h)

    def draw(self):
        rect = pg.Rect(self.pos_x, self.pos_y, self.size_w, self.size_h)
        pg.draw.rect(screen, self.COLOR, rect)

    def update(self, dt, H):
        for pu in self.activePowerUps[:]:
            pu.duration -= dt
            if pu.duration < 0:
                self.removePU(pu)
        super().update(dt)
        self.vertBounds(0,H)

    def removePU(self, pu):
        self.MAX_V_PADDLE-=pu.SPEED_P
        self.MAX_V_BALL-=pu.SPEED_B
        self.size_h-=pu.SIZE
        self.pos_y+=pu.SIZE/2
        self.activePowerUps.remove(pu)
        self.updateColor()
    
    def updateColor(self):
        if len(self.activePowerUps)==0:
            self.COLOR=(200,200,200)
        else:
            r,g,b=0,0,0
            for pu in self.activePowerUps:
                r=max(pu.COLOR[0],r)
                g=max(pu.COLOR[1],g)
                b=max(pu.COLOR[2],b)
            self.COLOR=(r,g,b)

    def reset(self, H):
        self.pos_y=H/2-self.size_h/2
        for pu in self.activePowerUps[:]:
            self.removePU(pu)
        

class Ball(GameObject):
    def __init__(self, x, y, w, h, V_0):
        self.COLOR = (200,200,200)
        super().__init__(x, y, w, h, V_0*random.choice([-1, 1]), V_0*random.uniform(-0.5,0.5))

    def interactPaddle(self, paddle):
        if self.totalOverlap(paddle):
            self.COLOR=paddle.COLOR
            hit_y = self.pos_y + self.size_h / 2
            paddle_center = paddle.pos_y + paddle.size_h / 2
            self.speed_y = paddle.MAX_V_BALL * (2 * (hit_y - paddle_center) / paddle.size_h)
            if self.speed_x<0:
                self.leftBounds(paddle.pos_x+paddle.size_w)
                self.speed_x = paddle.MAX_V_BALL
            elif self.speed_x>0:
                self.rightBounds(paddle.pos_x)
                self.speed_x = -paddle.MAX_V_BALL

    def draw(self):
        rect = pg.Rect(self.pos_x, self.pos_y, self.size_w, self.size_h)
        pg.draw.rect(screen, self.COLOR, rect)

    def update(self, dt, H):
        super().update(dt)
        self.vertBounds(0,H)

    def reset(self, H, W, V_0):
        self.COLOR=(200,200,200)
        self.pos_x=W/2
        self.pos_y=H/2
        self.speed_x=V_0*random.choice([-1, 1])
        self.speed_y=V_0*random.uniform(-0.5,0.5)

        

pg.init()

# Constants
WIDTH, HEIGHT = 1200, 800
SCORE_HEIGHT = 100
FPS = 120
PADDLE_DIST = 50
PADDLE_H = 50
PADDLE_W = 10
BALL_H = 8
BALL_W = 8
PU_H = 32
PU_W = 32
PU_V = 300    
PU_T = 15       #How long do power ups last
PU_G = 2       #How often on avg new powerups spawn
BALL_V = 300
PADDLE_V = 200

# Colors
C_BACKGROUND = (0, 0, 0)

# Window setup
screen = pg.display.set_mode((WIDTH, HEIGHT+SCORE_HEIGHT))
pg.display.set_caption("Pong")
font = pg.font.Font(None, 64)

# Objects
paddle1 = Paddle(PADDLE_DIST,HEIGHT/2-PADDLE_H/2,PADDLE_W, PADDLE_H,PADDLE_V,BALL_V)
paddle2 = Paddle(WIDTH-PADDLE_DIST,HEIGHT/2-PADDLE_H/2,PADDLE_W, PADDLE_H,PADDLE_V,BALL_V)
ball = Ball(WIDTH/2-BALL_W/2,HEIGHT/2-BALL_H/2,BALL_W,BALL_H,BALL_V)

paddles = [paddle1, paddle2]
powerUps = []

# Main loop
running = True
clock = pg.time.Clock()
startTime=pg.time.get_ticks()
prevTime=startTime
while running:
    current_time = pg.time.get_ticks()  
    dt = (current_time - prevTime) / 1000

    # Event handling
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    keys = pg.key.get_pressed()
    if keys[pg.K_UP]:
        paddle2.speed_y=-paddle2.MAX_V_PADDLE
        if keys[pg.K_DOWN]:
            paddle2.speed_y=0
    elif keys[pg.K_DOWN]:
        paddle2.speed_y=paddle2.MAX_V_PADDLE
    else:
        paddle2.speed_y=0
    if keys[pg.K_w]:
        paddle1.speed_y=-paddle1.MAX_V_PADDLE
        if keys[pg.K_s]:
            paddle1.speed_y=0
    elif keys[pg.K_s]:
        paddle1.speed_y=paddle1.MAX_V_PADDLE
    else:
        paddle1.speed_y=0

    # Interactions
    for p in paddles:
        ball.interactPaddle(p)
        for pu in powerUps[:]:  
            if pu.interactPaddle(p):
                powerUps.remove(pu)

    # Check game end
    scored=False
    if ball.leftOverlap(0):
        scored=True
        paddle2.score+=1
    if ball.rightOverlap(WIDTH):
        scored=True
        paddle1.score+=1
    if scored:
        ball.reset(HEIGHT,WIDTH,BALL_V)
        for p in paddles:
            p.reset(HEIGHT)
        powerUps.clear()

    # Update positions
    for p in paddles:
        p.update(dt, HEIGHT)
    ball.update(dt,HEIGHT)
    for pu in powerUps:
        pu.update(dt,HEIGHT,WIDTH)

    # Generate power ups
    u = random.random()
    t = -PU_G*math.log(u)
    if t<dt:
        powerUps.append(PowerUp(WIDTH/2,HEIGHT*random.random(),PU_W,PU_H,PU_V,PU_T))

    # Clear screen
    screen.fill(C_BACKGROUND)  

    # Draw 
    ball.draw()
    for p in paddles:
        p.draw()
    for pu in powerUps:
        pu.draw()

    rect1 = pg.Rect(0, HEIGHT, WIDTH/2-1, SCORE_HEIGHT)
    pg.draw.rect(screen, (255,255,255), rect1)
    text1 = font.render(f"{paddle1.score}", True, (0,0,0))
    text_rect1 = text1.get_rect(center=rect1.center)
    screen.blit(text1, text_rect1)
    
    rect2 = pg.Rect(WIDTH/2, HEIGHT, WIDTH/2, SCORE_HEIGHT)
    pg.draw.rect(screen, (255,255,255), rect2)
    text2 = font.render(f"{paddle2.score}", True, (0,0,0))
    text_rect2 = text2.get_rect(center=rect2.center)
    screen.blit(text2, text_rect2)   

    pg.display.flip() 
    
    # Game speed
    clock.tick(FPS)
    prevTime = current_time
    if scored:
        pg.time.delay(500)
        prevTime = pg.time.get_ticks()

pg.quit()