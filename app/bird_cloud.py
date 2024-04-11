import pygame
from app.bird import Bird

class BirdCloud(pygame.sprite.Sprite):
    def __init__(self, window, n_birds):
        super().__init__()
        self.window = window
        self.n_birds = n_birds
        
        # Creating the list containing the birds
        self.bird_list = []
        for i in range(self.n_birds):
            new_bird = Bird(window=self.window, speed=5, init_height=self.window.get_height() - 150)
            self.bird_list.append(new_bird)
        
    def fly(self):
        for bird in self.bird_list:
            bird.fly()
    
    def draw(self):
        for bird in self.bird_list:
            bird.draw()

    def flap(self, bird_index=None):
        if bird_index is not None:
            self.bird_list[bird_index].flap()
        else:
            for bird in self.bird_list:
                bird.flap()