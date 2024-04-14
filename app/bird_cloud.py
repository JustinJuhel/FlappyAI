import pygame
import numpy as np

class BirdCloud(pygame.sprite.Sprite):
    def __init__(self, window, n_birds, horiz_speed):
        super().__init__()
        self.window = window
        self.n_birds = n_birds
        
        init_x, init_y, init_vy, init_is_alive = 200, 450, -30, 1

        self.array = np.array([[init_x, init_y, init_vy, init_is_alive] for _ in range(n_birds)])
        self.bird_color = (255, 255, 255)
        self.bird_diameter = 20
        self.gravity = 5

        self.flap_force = -30
        self.vy_max = - 5 * self.flap_force
        self.gravity = 5
        self.horiz_speed = horiz_speed

        self.genome_size = 240
        self.genome = np.array((self.n_birds, self.genome_size))
    
    def draw(self):
        for params in self.array:
            position = (params[0], params[1])
            pygame.draw.circle(self.window, self.bird_color, position, self.bird_diameter)
    
    def fly(self):
        life_and_decent_speed_condition = [(params[3] == 1 and params[2] >= - self.vy_max) for params in self.array]
        self.array[:, 2] = np.where(
            life_and_decent_speed_condition,#self.array[:, 3] & self.array[:, 2] >= self.vy_max, # condition (if bird alive and max speed not reached yet)
            self.array[:, 2] + self.gravity, # new value if condition is True (modify speed)
            self.array[:, 2] # new value if condition is False
            )

        life_condition = [(elt == 1) for elt in self.array[:, 3]]
        self.array[:, 1] = np.where(
            life_condition, # if bird alive
            self.array[:, 1] + self.array[:, 2], # (modify y)
            self.array[:, 1]
        )

        death_condition = [(elt == 0) for elt in self.array[:, 3]]
        self.array[:, 0] = np.where(
            death_condition,
            self.array[:, 0] - self.horiz_speed,
            self.array[:, 0]
        )

        self.draw()

    def flap(self, index):
        if not isinstance(index, list): # Making sure we work with a list (even of length 1)
            index = [index]

        for i in index: # Flapping some birds
            self.array[i, 2] = self.flap_force

    def kill(self, frontier): # frontier is a list of positions: [(x1, y1), (x2, y2), ...]
        def dist(bird, point): # bird is structured as [x, y, vy, alive]
            return np.sqrt((bird[0] - point[0]) ** 2 + (bird[1] - point[1]) ** 2)

        alive_and_touch_frontier_condition = [(bird[3] and np.array([(dist(bird, point) <= self.bird_diameter) for point in frontier]).any()) for bird in self.array]
        self.array[:, 3] = np.where(
            alive_and_touch_frontier_condition,
            0,
            self.array[:, 3]
        )
    
    def all_birds_dead(self):
        return not self.array[:, 3].any()
    
    # def accelerate(self, acceleration):
    #     self.horiz_speed += acceleration