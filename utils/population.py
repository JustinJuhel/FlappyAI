import numpy as np
# Our Modules
from utils.network import BirdBrain

class Population:
    def __init__(self, n_birds, n_reproducers, genomes=None):
        self.n_birds = n_birds
        self.genomes = genomes

        # Group of birds
        init_x, init_y, init_vy, init_is_alive = 200, 450, -30, 1
        self.birds = np.array([[init_x, init_y, init_vy, init_is_alive] for _ in range(n_birds)])

        self.bird_diameter = 20
        self.gravity = 5

        self.flap_force = -30
        self.vy_max = - 5 * self.flap_force
        self.gravity = 5
        self.horiz_speed = 7

        print("Population -> __init__ : initialising the brains of the birds")
        self.genome_size = 200
        if genomes is not None:
            self.brains = np.array([BirdBrain(batch_size=n_birds, genome=genome) for genome in self.genomes])
        else:
            # self.brains = np.array([BirdBrain(batch_size=n_birds, genome=None) for _ in range(self.genome_size)])
            self.brains = np.array([BirdBrain(batch_size=n_birds, genome=None) for _ in range(n_birds)])
            self.genomes = np.array([brain.genome for brain in self.brains])

        self.collided_birds = [0 for bird in self.birds]
        self.collision_printed = [False for bird in self.birds]

        self.n_reproducers = n_reproducers
        # We start by taking n_reproducers random different birds from the available birds.
        self.reproducers = np.random.choice(np.arange(n_birds), size=self.n_reproducers, replace=False)


    def fly(self):
        alive_and_decent_speed = [(params[3] == 1 and params[2] >= - self.vy_max) for params in self.birds]
        self.birds[:, 2] = np.where(
            alive_and_decent_speed, # condition (if bird alive and max speed not reached yet)
            self.birds[:, 2] + self.gravity, # new value if condition is True (modify speed)
            self.birds[:, 2] # new value if condition is False
            )

        alive = [(elt == 1) for elt in self.birds[:, 3]]
        self.birds[:, 1] = np.where(
            alive, # if bird alive
            self.birds[:, 1] + self.birds[:, 2], # (modify y)
            self.birds[:, 1]
        )

        dead = [(elt == 0) for elt in self.birds[:, 3]]
        self.birds[:, 0] = np.where(
            dead,
            self.birds[:, 0] - self.horiz_speed,
            self.birds[:, 0]
        )

        for bird in self.birds:
            height = bird[1]
            if height < 0 or height > 600:
                print(f"""Bird escaped from the box.
                      x = {bird[0]}
                      y = {bird[1]}
                      vy = {bird[2]}
                      alive = {bird[3]}""")

    def flap(self, index):
        if not isinstance(index, list): # Making sure we work with a list (even of length 1)
            index = [index]

        for i in index: # Flapping some birds
            self.birds[i, 2] = self.flap_force

    def kill(self, global_frontier, floor_frontier, ceiling_frontier): # frontier is a list of positions: [(x1, y1), (x2, y2), ...]
        # Function to get the distance from a bird to a given point.
        def dist(bird, point): # bird is structured as [x, y, vy, alive]
            distance = np.sqrt((bird[0] - point[0]) ** 2 + (bird[1] - point[1]) ** 2)
            return distance
        # Function to get the birds that are collided at the current state.
        def get_collided_birds(global_frontier=global_frontier, floor_frontier=floor_frontier, ceiling_frontier=ceiling_frontier):
            for i, bird in enumerate(self.birds):
                x, y = bird[0], bird[1]
                # Looking if the bird is near a frontier point
                if np.array([(dist(bird, point) <= self.bird_diameter) for point in global_frontier]).any():
                    self.collided_birds[i] = 1
                    if not self.collision_printed[i]:
                        self.collision_printed[i] = True
                        print(f"Bird {i} collided. Last Position: {(x, y)}. Cause: Collision")
                # To be sure, verifying that the bird didn't escape from the box
                if y > floor_frontier[1] or y < ceiling_frontier[1]:
                    self.collided_birds[i] = 1
                    if not self.collision_printed[i]:
                        self.collision_printed[i] = True
                        print(f"Bird {i} collided. Last Position: {(x, y)}. Cause: Escaped from box")
            return self.collided_birds
        # Creating a condition: the birds that are alive but also collided (and will die)
        alive = [bird[3] for bird in self.birds]
        collided_birds = get_collided_birds()
        alive_and_collision = alive and collided_birds
        # Killing the chosen birds
        self.birds[:, 3] = np.where(
            alive_and_collision,
            0,
            self.birds[:, 3]
        )
        # Updating the reproducers
        birds_alive = np.where(self.birds[:, 3] == 1)[0]  # Getting the indices of the birds which are still alive
        if len(birds_alive) >= self.n_reproducers:  # If there is n_reproducers or more remaining birds, we take random birds among them
            self.reproducers = np.random.choice(birds_alive, size=self.n_reproducers, replace=False)
        # We return the reproducers to have access to them in the main program
        return self.reproducers
    
    def all_birds_dead(self):
        return not self.birds[:, 3].any()