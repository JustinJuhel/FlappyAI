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

        self.frontier = []

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
        vertical_frontier =[(self.x, y) for y in range(self.height)] + [(self.x, y) for y in range(self.height + self.gap, self.window.get_height())] # vertic. front. of upper + lower pipes
        horiz_frontier_up = [(x, y) for x in range(self.x, self.x + self.width) for y in [self.height]]
        horiz_frontier_down = [(x, y) for x in range(self.x, self.x + self.width) for y in [self.height + self.gap]]

        min_vert_fr = [vertical_frontier[i] for i in range(len(vertical_frontier)) if i % 10 == 0]
        min_horiz_fr_up = [horiz_frontier_up[i] for i in range(len(horiz_frontier_up)) if i % 10 == 0]
        min_horiz_fr_down = [horiz_frontier_down[i] for i in range(len(horiz_frontier_down)) if i % 10 == 0]

        self.frontier = min_horiz_fr_up + min_horiz_fr_down + min_vert_fr