import pygame

class Floor(pygame.sprite.Sprite):
    def __init__(self, window, height):
        super().__init__()
        self.height = height
        self.window = window

        self.color = (255, 255, 255)

        # self.frontier = [(x, self.window.get_height() - self.height) for x in range(self.window.get_width())]
        self.frontier = [(200, self.window.get_height() - self.height)]
    
    def draw(self):
        pygame.draw.rect(self.window, self.color, pygame.Rect(0, self.window.get_height() - self.height, self.window.get_width(), self.height)) # x, y, width, height