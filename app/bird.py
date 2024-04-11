import pygame

class Bird(pygame.sprite.Sprite):
    def __init__(self, window, speed, init_height):
        super().__init__()
        self.window = window
        self.speed = speed
        # self.init_height = init_height # the initial height at each flap
        self.height = init_height # the current height
        

        self.color = (255, 255, 255)
        self.x = 200
        self.diameter = 20

        self.flap_force = -30

        self.vy = self.flap_force
        self.vy_max = - 5 * self.flap_force
        self.gravity = 5

        self.pos = (self.x, self.height)

    def draw(self):
        pygame.draw.circle(self.window, self.color, self.pos, self.diameter)

    def fly(self):
        if self.vy >= - self.vy_max:
            self.vy += self.gravity
        self.height += self.vy
        self.pos = (self.x, self.height)
        self.draw()  # draw the bird at its new position

    def flap(self):
        self.vy = self.flap_force # just reinitializing the vertical speed

    def accelerate(self, acceleration):
        self.speed += acceleration