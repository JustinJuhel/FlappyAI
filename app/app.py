import pygame
import sys
import random as rd

from .pipe import Pipe

class App:
    def __init__(self):
        pass

    def launch(self):
        # Initializing PyGame
        pygame.init()
        # Creating a window
        window = pygame.display.set_mode((800, 600))

        # Instantiating our objects
        pipe_speed = 5
        pipe = Pipe(window=window, height=300)
        pipes = [pipe]
        dt = 40 # ms
        time_since_last_pipe = 0
        dist_between_pipes = 300

        # PyGame events loop
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
            window.fill((0, 0, 0))

            # Moving all our pipes and cancelling the ones that are out of screen
            for pipe in pipes:
                pipe.move()

            #     if pipe.x < - pipe.width:
            #         pipes.remove(pipe)
            #         pipe.kill()

            dist_since_last_pipe = (pipe_speed / dt) * time_since_last_pipe
            if dist_since_last_pipe >= dist_between_pipes:
                new_pipe_height = rd.randint(50, window.get_height() - 50)
                new_pipe = Pipe(window=window, height=new_pipe_height)
                pipes.append(new_pipe)
                time_since_last_pipe = 0

            
            pygame.display.flip() # Updating the window display
            pygame.time.wait(dt) # Making the app run at 25 images per second
            time_since_last_pipe += dt
            