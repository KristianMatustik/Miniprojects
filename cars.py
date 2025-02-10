import math
import pygame as pg


# Car game, will try teaching it to drive itself (prob. basic NN and evolution alg. as I have seen somewhere), so far implementing the basic game

class Track:
    def __init__(self, image_path, width, height):
        self.image = pg.image.load(image_path)
        self.image = pg.transform.scale(self.image, (width, height))
        self.array = pg.surfarray.array3d(self.image)
        self.width = width
        self.height = height

    def draw(self, screen):
        screen.blit(self.image, (0, 0))

class Car:
    def __init__(self, x, y, w, h, a, v):
        self.pos_x = x
        self.pos_y = y
        self.__speed = 0
        self.__angle = 0
        self.speed = v
        self.angle = a
        self.size_w = w
        self.size_h = h
        self.image = pg.image.load('img/car.png')
        self.image = pg.transform.scale(self.image, (w, h))

    @property
    def speed(self):
        return self.__speed
    @speed.setter
    def speed(self, val):
        self.__speed = val
        self.__speed_x = val * math.sin(math.radians(self.angle))
        self.__speed_y = -val * math.cos(math.radians(self.angle))

    @property
    def angle(self):
        return self.__angle
    @angle.setter
    def angle(self, val):
        self.__angle = val
        self.__speed_x = self.__speed * math.sin(math.radians(val))
        self.__speed_y = -self.__speed * math.cos(math.radians(val))

    def turn(self, angle):
        self.angle += angle
    
    def accelerate(self, speed):
        self.speed += speed

    def update(self, dt):
        self.pos_x += self.__speed_x * dt
        self.pos_y += self.__speed_y * dt

    def draw(self, screen):
        rotated_image = pg.transform.rotate(self.image, -self.angle)  # Rotate clockwise
        new_rect = rotated_image.get_rect(center=(self.pos_x, self.pos_y))
        screen.blit(rotated_image, new_rect.topleft)

    def distance_to_edge(self, track, angle_offset=0):
        track_array = track.array
        radius = 0.5 * self.size_h
        step = 5
        distance = 0
        ang = math.radians(self.angle + angle_offset)

        while True:
            check_x = int(self.pos_x + (radius + distance) * math.sin(ang))
            check_y = int(self.pos_y - (radius + distance) * math.cos(ang))
            if check_x < 0 or check_x >= track.width or check_y < 0 or check_y >= track.height:
                break
            if (track_array[check_x, check_y] == [0, 0, 0]).all():
                break
            distance += step
        return distance

    def distance_front(self, track):
        return self.distance_to_edge(track, 0)

    def distance_left(self, track):
        return self.distance_to_edge(track, -30)

    def distance_right(self, track):
        return self.distance_to_edge(track, 30)

    def check_collision(self, track):
        track_array = track.array
        radius = 0.5 * self.size_h

        for offset in [-30, 0, 30]:
            ang = math.radians(self.angle + offset)
            gx = int(self.pos_x + radius * math.sin(ang))
            gy = int(self.pos_y - radius * math.cos(ang))
            if gx < 0 or gx >= track.width or gy < 0 or gy >= track.height:
                return True
            if (track_array[gx, gy] == [0, 0, 0]).all():
                return True

        return False

car = Car(100, 200, 50, 100, 0, 0)
car.angle=90

pg.init()

# Constants
WIDTH, HEIGHT = 1200, 800
FPS = 30

# Colors
C_BACKGROUND = (255, 255, 255)

# Window setup
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Cars")
font = pg.font.Font(None, 64)

# Create track
track = Track('img/track.png', WIDTH, HEIGHT)

# Main loop
running = True
clock = pg.time.Clock()
startTime = pg.time.get_ticks()
prevTime = startTime
while running:
    current_time = pg.time.get_ticks()
    dt = (current_time - prevTime) / 1000

    # Event handling
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_UP:
                car.accelerate(5)
            if event.key == pg.K_DOWN:
                car.accelerate(-5)
            if event.key == pg.K_LEFT:
                car.turn(-5)
            if event.key == pg.K_RIGHT:
                car.turn(5)
    # keys = pg.key.get_pressed()
    # if keys[pg.K_LEFT]:
    #     car.turn(-5 * dt)
    # if keys[pg.K_RIGHT]:
    #     car.turn(5 * dt)
    # if keys[pg.K_UP]:
    #     car.accelerate(5 * dt)
    # if keys[pg.K_DOWN]:
    #     car.accelerate(-5 * dt)

    # Interactions
    front_distance = car.distance_front(track)
    left_distance = car.distance_left(track)
    right_distance = car.distance_right(track)
    print(f"Front: {front_distance}, Left: {left_distance}, Right: {right_distance}")

    if car.check_collision(track):
        print("Collision detected!")


    # Check game end
    #collision

    # Update positions
    car.update(dt)

    # Draw
    track.draw(screen)  # Draw the track
    car.draw(screen)

    pg.display.flip()
    
    # Game speed
    clock.tick(FPS)
    prevTime = current_time

pg.quit()