import pygame
from app.bird import Bird

class BirdCloud(pygame.sprite.Sprite):
    def __init__(self, window, n_birds):
        super().__init__()
        self.window = window
        self.n_birds = n_birds
        
        # Creating the list containing the birds
        self.birds = []
        for i in range(self.n_birds):
            new_bird = Bird(window=self.window, speed=5, init_height=self.window.get_height() - 150)
            self.birds.append(new_bird)
        
    def fly(self):
        for bird in self.birds:
            bird.fly()
    
    def draw(self):
        for bird in self.birds:
            bird.draw()

    def flap(self, bird_index=None):
        if bird_index is not None:
            self.birds[bird_index].flap()
        else:
            for bird in self.birds:
                bird.flap()