import math
import pygame as pg
import numpy as np
import cProfile
import pstats
import pickle


# Reinforcment learning, evolution algorithm to teach NNs to drive cars around a track

class Track:
    def __init__(self, image_path, width, height):
        self.image = pg.image.load(image_path)
        self.image = pg.transform.scale(self.image, (width, height))
        self.array = pg.surfarray.array3d(self.image)
        self.width = width
        self.height = height
        self.mask = np.all(self.array == [0, 0, 0], axis=2)

    def draw(self, screen):
        screen.blit(self.image, (0, 0))

class Car:
    def __init__(self, x, y, v, a, w, h ):
        self.__speed = v
        self.__angle = a
        self.set(x, y, v, a)
        self.size_w = w
        self.size_h = h

        self.max_accel = 300
        self.max_turn = 90

        self.image = pg.image.load('img/car.png')
        self.image = pg.transform.scale(self.image, (w, h))

    def set(self, x, y, v, a):
        self.pos_x = x
        self.pos_y = y
        self.speed = v
        self.angle = a

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
        self.speed = self.speed

    def turn(self, a, dt):
        self.angle += a*self.max_turn*dt
    
    def accelerate(self, a, dt):
        self.speed += a*self.max_accel*dt

    def update(self, dt):
        self.pos_x += self.__speed_x * dt
        self.pos_y += self.__speed_y * dt

    def draw(self, screen):
        rotated_image = pg.transform.rotate(self.image, -self.angle)
        new_rect = rotated_image.get_rect(center=(self.pos_x, self.pos_y))
        screen.blit(rotated_image, new_rect.topleft)

        for offset in [-30, 0, 30]: # Lines representing crash detection and distance messurement
            angle_rad = math.radians(self.angle + offset)
            red_line_length = self.size_h//2
            blue_line_length = self.size_h//2

            red_end_x = self.pos_x + red_line_length * math.sin(angle_rad)
            red_end_y = self.pos_y - red_line_length * math.cos(angle_rad)
            pg.draw.line(screen, (255, 0, 0), (self.pos_x, self.pos_y), (red_end_x, red_end_y), 2)

            blue_end_x = red_end_x + blue_line_length * math.sin(angle_rad)
            blue_end_y = red_end_y - blue_line_length * math.cos(angle_rad)
            pg.draw.line(screen, (0, 0, 255), (red_end_x, red_end_y), (blue_end_x, blue_end_y), 2)

    # This method takes most time, should be optimized (maybe route from discrete lines, iterate them and calculate intersection)
    # This implementation allows simple track specification though, simple B-W image
    def distance_to_edge(self, track, angle_offset=0):
        radius = 0.5 * self.size_h
        step = 5    # can be sped up by longer step, NN less acurate though, and can ignore thin walls
        distance = 0
        ang = math.radians(self.angle + angle_offset)

        while True:
            check_x = int(self.pos_x + (radius + distance) * math.sin(ang))
            check_y = int(self.pos_y - (radius + distance) * math.cos(ang))
            if check_x < 0 or check_x >= track.width or check_y < 0 or check_y >= track.height:
                break
            if track.mask[check_x, check_y]:
                break
            distance += step
        return distance

    def distance_front(self, track):
        return self.distance_to_edge(track, 0)

    def distance_left(self, track):
        return self.distance_to_edge(track, -30)

    def distance_right(self, track):
        return self.distance_to_edge(track, 30)

    def check_collision(self, track):   #only at the front, same points as distance messurement (at d=0), enough for learning
        radius = 0.5 * self.size_h

        for offset in [-30, 0, 30]:
            ang = math.radians(self.angle + offset)
            gx = int(self.pos_x + radius * math.sin(ang))
            gy = int(self.pos_y - radius * math.cos(ang))
            if gx < 0 or gx >= track.width or gy < 0 or gy >= track.height:
                return True
            if track.mask[gx, gy]:
                return True

        return False

class Driver:
    def __init__(self):
        self.W1 = np.random.uniform(-0.5, 0.5, (4, 5))
        self.b1 = np.random.uniform(-0.5, 0.5, (1, 5))
        self.W2 = np.random.uniform(-0.5, 0.5, (5, 2))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, 2))

    def adjust_controls(self, track, car, dt):
        d_left = car.distance_left(track)
        d_front = car.distance_front(track)
        d_right = car.distance_right(track)
        speed = car.speed
        x = np.array([[d_left, d_front, d_right, speed]])

        hidden = np.tanh(x @ self.W1 + self.b1)
        out = np.tanh(hidden @ self.W2 + self.b2)

        accel = out[0, 0]
        turn_amount = out[0, 1]
        car.accelerate(accel, dt)
        car.turn(turn_amount, dt)

    def mutate(self, scale=0.1):
        self.W1 += np.random.normal(0, scale, self.W1.shape)
        self.b1 += np.random.normal(0, scale, self.b1.shape)
        self.W2 += np.random.normal(0, scale, self.W2.shape)
        self.b2 += np.random.normal(0, scale, self.b2.shape)
        return self 

    def copy(self):
        new_driver = Driver()
        new_driver.W1 = self.W1.copy()
        new_driver.b1 = self.b1.copy()
        new_driver.W2 = self.W2.copy()
        new_driver.b2 = self.b2.copy()
        return new_driver

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

# Main method, runs sim, drawing to window if FPS is set, manual controls if no Driver specified
# Returns distance traveled as fitness score for AI learning
def run_sim(track, car, fps=math.inf, DT=-1, car_driver=None, max_time=math.inf):
    clock = pg.time.Clock()
    distance_traveled = 0
    visited_positions = []
    old_x, old_y = car.pos_x, car.pos_y

    running = True
    start_time = pg.time.get_ticks()
    prev_time = start_time
    while running:
        current_time = pg.time.get_ticks()

        # Calculate dt, use fixed specified value for fast training or sped up drawing
        if (DT == -1):
            dt = (current_time - prev_time) / 1000
        else:
            dt = DT
        max_time -= dt

        # Check end, run out of time or crash
        if (max_time!=math.inf and max_time< 0) or car.check_collision(track):
            running = False
            break

        # Check window closed. Need to fix so it doesn't crash on exit
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return 0  

        # Car controls, manual or autopilot "driver"
        if car_driver is None:
            keys = pg.key.get_pressed()
            if keys[pg.K_LEFT]:
                car.turn(-1, dt)
            if keys[pg.K_RIGHT]:
                car.turn(1, dt)
            if keys[pg.K_UP]:
                car.accelerate(1, dt)
            if keys[pg.K_DOWN]:
                car.accelerate(-1, dt)
        else:
            car_driver.adjust_controls(track, car, dt)

        # Calculate distance traveled (fitness score for AI), prevents counting driving in circles
        new_x, new_y = car.pos_x, car.pos_y
        segment_dist = math.dist((old_x, old_y), (new_x, new_y))
        if all(math.dist((new_x, new_y), (vx, vy)) > 50 for vx, vy in visited_positions) and car.speed > 0:
            distance_traveled += segment_dist
            visited_positions.append((new_x, new_y))
            old_x, old_y = new_x, new_y
        # Clear visited_positions after some size, speeds up calculation, allows more laps of circuit
        if len(visited_positions) > 40:
            visited_positions = visited_positions[len(visited_positions)//2:]

        # Update car position
        car.update(dt)

        # Draw (if FPS is set, else just simulation - for training)
        if fps != math.inf:
            track.draw(screen)
            car.draw(screen)
            pg.display.flip()
            clock.tick(fps)
        prev_time = current_time

    # Evaluate score, distance travel, also good to incorporate speed/time left somehow if the track is not circuit
    return distance_traveled


## Main program
# Setup
pg.init()

WIDTH, HEIGHT = 1200, 800
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Cars")

track = Track('img/track.png', WIDTH, HEIGHT)
car_x, car_y = 50, 600
car_w, car_h = 25, 50
car_v, car_a = 10, 0
car = Car(car_x, car_y, car_v, car_a, car_w, car_h)

FPS=30

# Training specs
num_drivers = 100
num_generations = 50
time_limit = 30 # lower time trains faster, need bit higher to complete whole track
dt_train = 0.1 # higher trains faster, but can be inferior when used with other lower dt not trained on (eg. playing real speed 50FPS. max~0.2)

# Manual controls ride
#run_sim(track, car, FPS, -1, None, time_limit)

# Watch the best driver
# best_driver = Driver.load('files/best_driver.pkl') # trained with 100/100/45/0.02 (best with 50 FPS) fitness 12017
# car.set(car_x, car_y, car_v, car_a)
# fitness = run_sim(track, car, 50, 0.02, best_driver) #increase FPS from 50 to replay faster

# Evolution algorithm
# with cProfile.Profile() as pr:
drivers = [Driver() for _ in range(num_drivers)]
best_drivers=[]
for g in range(num_generations):
    results = []
    for driver in drivers:
        car.set(car_x, car_y, car_v, car_a)
        dt=dt_train
        #Adaptable dt for training - higher is fast to train, but models can be inacurate when run with smaller dt, better train it on the one used (drivers "overfitted" to track and parameters)
        #dt = dt_train - ((dt_train-1/FPS) * g / num_generations)
        fitness = run_sim(track, car, math.inf, dt, driver, time_limit)
        results.append((driver, fitness))

    # Sort from best
    results.sort(key=lambda x: x[1], reverse=True)

    # Print stats
    best_distance = results[0][1]
    avg_distance = sum(r[1] for r in results) / num_drivers
    print(f"Generation: {g}")
    print(f"Avg: {avg_distance}")
    print(f"Best: {best_distance}")

    #save best driver from each generation
    best_drivers.append(results[0][0])

    # Last generation just sort by fitness
    if g == num_generations - 1:
        drivers = [r[0] for r in results]
        break

    # Other gens mutate, 20 % best are kept + 2x mutated, 20 % next best also kept + 1x mutated
    n = num_drivers // 5
    scale = 0.1*(num_generations-g)/num_generations
    new_drivers = []
    for gen in range(n):
        new_drivers.append(results[gen][0])
        new_drivers.append(results[gen][0].copy().mutate(scale))
        new_drivers.append(results[gen][0].copy().mutate(scale))

    for gen in range(n, 2*n):
        new_drivers.append(results[gen][0])
        new_drivers.append(results[gen][0].copy().mutate(scale))

    drivers = new_drivers
# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats()

# # Print fitness from last generation
# for i, driver in enumerate(drivers):
#     car.set(car_x, car_y, car_v, car_a)
#     fitness = run_sim(track, car, math.inf, dt_train, driver, time_limit)
#     print(f"Driver {i}: {fitness}")

# Play best driver from each generation ("first" gens will be played all, after that by steps (biggest progress at start))
first = 10
for gen in range(0, first):
    car.set(car_x, car_y, car_v, car_a)
    if gen!=0 and best_drivers[gen]==best_drivers[gen-1]:
        print(f"Driver {gen}: {fitness} (same as previous)")
        continue
    fitness = run_sim(track, car, FPS, dt_train, best_drivers[gen], time_limit) #plays at 3x speed (30fps but steps are 0.1 s)
    #fitness = run_sim(track, car, FPS, -1, driver, time_limit) #can try running model at different dt, driver might crash though, overfitted to training, taking corners with 0 reserve
    print(f"Driver {gen}: {fitness}")

step=5
for gen in range(first, num_generations, step):
    car.set(car_x, car_y, car_v, car_a)
    if gen!=0 and best_drivers[gen]==best_drivers[gen-step]:
        print(f"Driver {gen}: {fitness} (same as previous)")
        continue
    fitness = run_sim(track, car, FPS, dt_train, best_drivers[gen], time_limit) #plays at 3x speed (30fps but steps are 0.1 s)
    #fitness = run_sim(track, car, FPS, -1, driver, time_limit) #can try running model at different dt, driver might crash though, overfitted to training, taking corners with 0 reserve
    print(f"Driver {gen}: {fitness}")

# Play best driver
# best_driver.save('files/best_driver.pkl')
car.set(car_x, car_y, car_v, car_a)
fitness = run_sim(track, car, math.inf, dt_train, drivers[0], time_limit) 
print(f"Best Driver ({num_generations-1}): {fitness}")
car.set(car_x, car_y, car_v, car_a)
run_sim(track, car, FPS, dt_train, drivers[0]) 
print("(Riding with no time limit)")

pg.quit()