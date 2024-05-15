from app.app import App
import random as rd
import itertools
import numpy as np

from utils.population import Population
from utils.pipe import Pipe

def main():

    # app = App()
    # app.launch()

    window_width, window_height = 800, 600

    n_birds = 32 # Number of birds in our population
    n_reproducers = int(np.ceil(0.10 * n_birds)) # Number of birds that are allowed to reproduce
    print(f"n_reproducers {n_reproducers}")
    reproducers = []
    genomes = None # Initial genome of the birds will be randomly generated
    mutation_rate = 0.01

    dt = 40 # ms
    time_since_last_pipe = 0
    dist_between_pipes = 400
    floor_height = 50
    floor_frontier = (200, window_height - floor_height)
    ceiling_frontier = (200, 0)

    n_generations = 1
    for _ in range(n_generations):
        # Creating the Population
        print("Creating the Population")
        population = Population(n_birds=n_birds, genomes=genomes)
        # Generating the Pipes
        print("Generating the Pipes")
        pipe1 = Pipe(window_dim=(window_width, window_height), height=300)
        pipe_speed = pipe1.speed
        pipes = [pipe1]

        # Used just for not choosing several times the same reproducers
        reproducers_selected = False

        # Launching the Generation Simulation
        print("Launching the Generation Simulation")
        running = True
        while running:
            # Moving the pipes
            for pipe in pipes:
                pipe.move()
            # Generating new pipes (no acceleration of the pipes)
            dist_since_last_pipe = (pipe_speed / dt) * time_since_last_pipe
            if dist_since_last_pipe >= dist_between_pipes:
                new_pipe_height = rd.randint(0, window_height - 200 - floor_height)
                new_pipe = Pipe(window_dim=(window_width, window_height), height=new_pipe_height)
                pipes.append(new_pipe)
                time_since_last_pipe = 0
            # Making the birds fly
            population.fly()
            ###
            # MAKING THE BIRDS RANDOMLY FLAP FOR NOW, THEN WE WILL USE THE NEURAL NETWORK TO PILOT THE BIRDS
            # n_flaps = int(np.ceil(0.15 * n_birds))
            # birds_to_flap = np.random.choice(n_birds, size=n_flaps, replace=False)
            # population.flap(birds_to_flap)
            for i in range(n_birds):
                if rd.random() < 0.15: # 15% chance of flap
                    population.flap(index=i)
            ###

            # Collision
            global_frontier = [floor_frontier] + [ceiling_frontier] + list(itertools.chain(*[curr_pipe.frontier for curr_pipe in pipes])) # concatenating all the frontiers
            # print(f"frontier {frontier}")
            population.kill(global_frontier, floor_frontier, ceiling_frontier)

            # Getting the 10% best birds' indices
            if (sum(population.birds[:, 3]) == n_reproducers) and (not reproducers_selected):
                reproducers = np.where(population.birds[:, 3] == 1)[0]
                print(f"Selecting our reproducers: {reproducers}")
                reproducers_selected = True
                

            # Condition for stopping the simulation
            running = False if population.all_birds_dead() else True

            # making time run
            time_since_last_pipe += dt

        # Getting the best birds' genomes
        print("Getting the best birds' genomes")
        selected_genomes = [population.brains[reproducer_index].genome for reproducer_index in reproducers]
        print(f"selected_genomes: {selected_genomes}")
        print(f"selected_genomes {type(selected_genomes)} of {type(selected_genomes[0])} of shape {selected_genomes[0].shape}")

        # Creating n_birds new genomes by crossing over the selected genomes
        print("Creating n_birds new genomes by crossing over the selected genomes")
        new_genomes = []
        for i in range(n_birds):
            i1 = np.random.randint(0, len(selected_genomes))
            i2 = np.random.randint(0, len(selected_genomes))
            while i1 == i2: # making sure i1 and i2 are different
                i2 = np.random.randint(0, len(selected_genomes))
            
            genome_parent1 = selected_genomes[i1]
            genome_parent2 = selected_genomes[i2]

            crossover_point = np.random.randint(0, len(genome_parent1))

            genome_child = np.concatenate((genome_parent1[:crossover_point], genome_parent2[crossover_point:])) # creating a new genome
            new_genomes.append(genome_child)

        # Mutations on the newborns
        print("Mutations on the newborns")
        mutated_genomes = new_genomes.copy()
        for genome in mutated_genomes:
            for i in range(len(genome)):
                min_val, max_val = min(genome), max(genomes)
                if np.random.rand() < mutation_rate:
                    genome[i] = np.random.uniform(min_val, max_val) # changing the value of the gene
        
        # Creating the next generation
        print("Creating the next generation")
        genomes = mutated_genomes


















if __name__ == "__main__":
    main()