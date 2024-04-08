import pygame

class Pipe(pygame.sprite.Sprite):
    def __init__(self, window, height):
        super().__init__()
        self.x = window.get_width()
        self.height = height
        self.window = window
        
        self.gap = 100
        self.width = 100
        self.speed = 5
        self.color = (255, 255, 255)

    def move(self):
        self.x -= self.speed
        self.draw()

    def draw(self):
        if self.x > - self.width:
            window_height = self.window.get_height()
            # Drawing upper rectangle
            pygame.draw.rect(self.window, self.color, pygame.Rect(self.x, 0, 100, self.height)) # x, y, width, height
            # Drawinf lower rectangle
            pygame.draw.rect(self.window, self.color, pygame.Rect(self.x, self.height + self.gap, 100, window_height - (self.height + self.gap))) # x, y, width, height