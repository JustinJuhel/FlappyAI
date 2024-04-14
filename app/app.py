import pygame
import sys
import random as rd
import itertools
import numpy as np

from .pipe import Pipe
# from .bird import Bird
from .floor import Floor
from .bird_cloud import BirdCloud

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

        n_birds = 16
        # cloud = BirdCloud(window, n_birds=n_birds)
        cloud = BirdCloud(window, n_birds=n_birds, horiz_speed=pipe_speed)

        floor_height = 50
        floor = Floor(window=window, height=floor_height)
        game_running = False
        display_start_message = True
        game_over = False
        display_end_message = False
        ceiling_frontier = [(200, 0)]

        # Creating a start message
        start_font = pygame.font.SysFont("Arial", 36)
        start_text = start_font.render("Click or type Space to Start!", True, (0, 255, 0))

        start_text_rect = start_text.get_rect()
        start_text_rect.center = window.get_rect().center
        # Creating an end message
        end_font = pygame.font.SysFont("Arial", 36)
        end_text = end_font.render("GAME OVER!", True, (0, 255, 0))

        end_text_rect = end_text.get_rect()
        end_text_rect.center = window.get_rect().center






        # PyGame events loop
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    cloud.flap(index=range(n_birds))
                if event.type == pygame.MOUSEBUTTONDOWN and not game_running and not game_over:
                    cloud.flap(index=range(n_birds))
                    game_running = True
                    # display_start_message = False

            window.fill((0, 0, 0))

            floor.draw()

            # Moving all our pipes and cancelling the ones that are out of screen
            for pipe in pipes:
                if game_running:
                    pipe.move()
                else:
                    pipe.draw() # if game over, the pipe doesn't move but we still draw it

            # Generating new pipes
            if game_running:
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
            if game_running:
                cloud.fly()
            else:
                cloud.draw()

            # Collision
            frontier = floor.frontier + ceiling_frontier + list(itertools.chain(*[curr_pipe.frontier for curr_pipe in pipes])) # concatenating all the frontiers
            # Drawing the frontier in red
            for point in frontier:
                pygame.draw.circle(window, (255, 0, 0), point, 2)
            cloud.kill(frontier)
            if cloud.all_birds_dead():
                # print("All birds are dead")
                game_running = False
                # display_end_message = True
                game_over = True

            # Randomly flapping
            if game_running:
                # for bird in birds_cloud:
                for i in range(n_birds):
                    if rd.random() < 0.15: # 15% chance of flap
                        # print("flapping")
                        cloud.flap(index=i)
            # Performing actions depending on the key pressed
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE] and game_running:
                cloud.flap(index=range(n_birds))
            if keys[pygame.K_SPACE] and not game_running and not game_over:
                cloud.flap(index=range(n_birds))
                game_running = True
                # display_start_message = False

            pygame.display.flip() # Updating the window display
            pygame.time.wait(dt) # Making the app run at 25 images per second
            if game_running:
                time_since_last_pipe += dt
            