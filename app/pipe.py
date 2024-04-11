import pygame

class Pipe(pygame.sprite.Sprite):
    def __init__(self, window, height, speed):
        super().__init__()
        self.x = window.get_width()
        self.height = height
        self.window = window
        
        self.gap = 200
        self.width = 100
        self.speed = speed
        self.color = (255, 255, 255)

        self.frontier = [
            (self.x, y) for y in range(self.height)
            ] + [
                (self.x, y) for y in range(self.height + self.gap, self.window.get_height() - (self.height + self.gap))
                ] + [
                    (x, y) for x in range(self.x, self.x + self.width) for y in [self.height, self.height + self.gap]
                    ]

    def move(self):
        self.x -= self.speed
        self.update_frontier()
        self.draw()

    def draw(self):
        if self.is_current():
            window_height = self.window.get_height()
            # Drawing upper rectangle
            pygame.draw.rect(self.window, self.color, pygame.Rect(self.x, 0, self.width, self.height)) # x, y, width, height
            # Drawinf lower rectangle
            pygame.draw.rect(self.window, self.color, pygame.Rect(self.x, self.height + self.gap, self.width, window_height - (self.height + self.gap))) # x, y, width, height

    def accelerate(self, acceleration):
        self.speed += acceleration

    def is_current(self): # returns a boolean saying if the pipe is currently on the screen
        return (self.x > - self.width)
    
    def update_frontier(self):
        # Making sure the x & y are integers
        self.x = round(self.x)
        self.height = round(self.height)
        # Updating the frontier with the current x and height
        self.frontier = [
            (self.x, y) for y in range(self.height) # vertical frontier of the upper pipe
            ] + [
                (self.x, y) for y in range(self.height + self.gap, self.window.get_height()) # vertical frontier of the lower pipe
                ] + [
                    (x, y) for x in range(self.x, self.x + self.width) for y in [self.height, self.height + self.gap] # horizontal frontier of both pipes
                    ]