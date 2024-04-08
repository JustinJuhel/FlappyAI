import pygame

class Pipe(pygame.sprite.Sprite):
    def __init__(self, window, height, speed):
        super().__init__()
        self.x = window.get_width()
        self.height = height
        self.window = window
        
        self.gap = 100
        self.width = 100
        self.speed = speed
        self.color = (255, 255, 255)

    def move(self):
        self.x -= self.speed
        self.draw()

    def draw(self):
        if self.is_current():
            window_height = self.window.get_height()
            # Drawing upper rectangle
            pygame.draw.rect(self.window, self.color, pygame.Rect(self.x, 0, 100, self.height)) # x, y, width, height
            # Drawinf lower rectangle
            pygame.draw.rect(self.window, self.color, pygame.Rect(self.x, self.height + self.gap, 100, window_height - (self.height + self.gap))) # x, y, width, height

    def accelerate(self, acceleration):
        self.speed += acceleration

    def is_current(self): # returns a boolean saying if the pipe is currently on the screen
        return (self.x > - self.width)