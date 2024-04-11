import pygame
import sys
import random as rd
import itertools
import numpy as np

from .pipe import Pipe
from .bird import Bird
from .floor import Floor

def dist(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class App:
    def __init__(self):
        pass

    def launch(self):
        # Initializing PyGame
        pygame.init()
        # Creating a window
        window = pygame.display.set_mode((800, 600))

        # Instantiating our objects
        pipe = Pipe(window=window, height=300, speed=7)
        pipe_speed = pipe.speed
        acceleration = 0.3
        pipes = [pipe]
        dt = 40 # ms
        time_since_last_pipe = 0
        dist_between_pipes = 400

        bird = Bird(window=window, speed=5, init_height=window.get_height() - 150)

        floor_height = 50
        floor = Floor(window=window, height=floor_height)
        game_over = False

        # PyGame events loop
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    bird.flap()
                
            window.fill((0, 0, 0))

            floor.draw()

            # Moving all our pipes and cancelling the ones that are out of screen
            for pipe in pipes:
                if not game_over:
                    pipe.move()
                else:
                    pipe.draw() # if game over, the pipe doesn't move but we still draw it

            # Generating new pipes
            dist_since_last_pipe = (pipe_speed / dt) * time_since_last_pipe
            if dist_since_last_pipe >= dist_between_pipes:
                new_pipe_height = rd.randint(0, window.get_height() - 200 - floor_height)
                new_pipe = Pipe(window=window, height=new_pipe_height, speed=pipe_speed)
                pipes.append(new_pipe)
                time_since_last_pipe = 0
                
                # Accelerating to make the game harder and harder
                for pipe in pipes:
                    if pipe.is_current():
                        pipe.accelerate(acceleration)
                pipe_speed += acceleration
            
            # Drawing the bird
            if not game_over:
                bird.fly()
            else:
                bird.draw() # if game over, the bird doesn't move but we still draw it

            # Collision
            bird_position = bird.pos
            pygame.draw.circle(window, (255, 0, 0), bird_position, 5)
            frontier = floor.frontier + list(itertools.chain(*[curr_pipe.frontier for curr_pipe in pipes])) # concatenating all the frontiers
            # print(frontier)
            # Drawing the frontier in red
            for point in frontier:
                pygame.draw.circle(window, (255, 0, 0), point, 2)
            min_dist = min([dist(bird_position, pos) for pos in frontier]) # the minimal distance to a frontier
            if min_dist <= bird.diameter: # if there is a collision
                # print("/!\ COLLISION /!\ ")
                game_over = True


            # Performing actions depending on the key pressed
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE] and not game_over:
                bird.flap()

            
            pygame.display.flip() # Updating the window display
            pygame.time.wait(dt) # Making the app run at 25 images per second
            time_since_last_pipe += dt
            