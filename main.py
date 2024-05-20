from app.app import App
import random as rd
import itertools
import numpy as np
import time

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
    mutation_rate = 0.02

    dt = 40 # ms
    time_since_last_pipe = 0
    dist_between_pipes = 400
    floor_height = 50
    floor_frontier = (200, window_height - floor_height)
    ceiling_frontier = (200, 0)

    max_distances = []

    n_generations = 20
    for i in range(n_generations):
        print("#"*150)
        print(f"GENERATION {i}")
        print("#"*150)
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

        # Used to record the performance of the best bird of the current generation
        max_dist = 0

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
            # Updating the max_dist with the distance done after this time step
            max_dist += pipe_speed * dt

            ###
            # MAKING THE BIRDS RANDOMLY FLAP FOR NOW, THEN WE WILL USE THE NEURAL NETWORK TO PILOT THE BIRDS
            # n_flaps = int(np.ceil(0.15 * n_birds))
            # birds_to_flap = np.random.choice(n_birds, size=n_flaps, replace=False)
            # population.flap(birds_to_flap)
            '''
            for i in range(n_birds):
                if rd.random() < 0.15: # 15% chance of flap
                    population.flap(index=i)
            '''
            print("Making the Birds Flap")
            # print(population.brains.shape)
            # print(type(population.brains[0]))
            outputs = []
            # print(len(population.brains))
            # print(population.brains)
            for i, brain in enumerate(population.brains):
                input = population.birds[i,:]
                # print(f"###### Bird {i} ######")
                output = brain.get_response(input)
                # print(f"NN Response: {output}, type: {type(output)}")#, type of first elt: {type(output[0])}")
                outputs.append(output.detach().numpy()[0])  # output is a tensor with 1 elt so we detach it, convert to numpy and then get the only elt to get a float
            # print(f"outputs: {outputs}")
            flap_indices = [i for i, value in enumerate(outputs) if value > 0]
            # Some birds might be dead so we don't take them
            real_flap_indices = [ind for ind in flap_indices if population.birds[ind, 3]]
            print(f"{len(flap_indices)} Birds are selected to flap but {len(real_flap_indices)} are alive.")
            population.flap(index=real_flap_indices)
                
            ###

            # Collision
            global_frontier = [floor_frontier] + [ceiling_frontier] + list(itertools.chain(*[curr_pipe.frontier for curr_pipe in pipes])) # concatenating all the frontiers
            # print(f"frontier {frontier}")
            population.kill(global_frontier, floor_frontier, ceiling_frontier)

            # Getting the 10% best birds' indices
            # It can be possible to never have exactly n_reproducers alive at the same time.
            # If it is not the case I duplicate the chosen reproducers until I have the right number of reproducers.
            print(f"Nb Birds Alive: {sum(population.birds[:, 3])}")
            if (sum(population.birds[:, 3]) <= n_reproducers) and (not reproducers_selected):
                reproducers = np.where(population.birds[:, 3] == 1)[0]
                # If we have not chosen enough reproducers
                while len(reproducers) < n_reproducers:
                    print("Duplicating some reproducers")
                    duplicate_index = rd.randint(0, len(reproducers) - 1)
                    reproducer_to_duplicate = reproducers[duplicate_index]
                    reproducers = np.insert(reproducers, len(reproducers), reproducer_to_duplicate)
                print(f"Selecting our reproducers: {reproducers}")
                reproducers_selected = True                

            # Condition for stopping the simulation
            running = False if population.all_birds_dead() else True

            # making time run
            time_since_last_pipe += dt

        # Getting the best birds' genomes
        print("Getting the best birds' genomes")
        selected_genomes = [population.brains[reproducer_index].genome for reproducer_index in reproducers]
        print(f"Selected {len(selected_genomes)} Genomes.")
        # print(f"selected_genomes: {selected_genomes}")
        # print(f"selected_genomes {type(selected_genomes)} of {type(selected_genomes[0])} of shape {selected_genomes[0].shape}")

        # Creating n_birds new genomes by crossing over the selected genomes
        print("Creating n_birds new genomes by crossing over the selected genomes")
        new_genomes = []
        for i in range(n_birds):
            i1 = np.random.randint(0, len(selected_genomes))
            i2 = np.random.randint(0, len(selected_genomes))
            while i1 == i2: # making sure i1 and i2 are different
                i2 = np.random.randint(0, len(selected_genomes))
            # print(f"{i1} & {i2} will make a baby bird")
            
            genome_parent1 = selected_genomes[i1]
            genome_parent2 = selected_genomes[i2]

            # print(f"type(genome_parent1): {type(genome_parent1)}")
            # print(f"genome_parent1.keys() {genome_parent1.keys()}")
            # print(f"genome_parent1['0.bias'].shape {genome_parent1['0.bias'].shape}")

            # crossover_point = np.random.randint(0, len(genome_parent1))
            crossover_rate = rd.random()
            crossover_points = {k: int(crossover_rate * genome_parent1[k].size) for k in genome_parent1.keys()}
            # print(f"crossover_points {crossover_points}")

            # To create the genome of the child we take each layers of both parents and we take the weights
            # of the first parent's genome until the crossover point and we finish with the second parent's
            # genome.
            genome_child = {
                k: np.concatenate((
                    genome_parent1[k].flatten()[:crossover_points[k]],
                    genome_parent2[k].flatten()[crossover_points[k]:]
                )).reshape(genome_parent1[k].shape)
                for k in genome_parent1.keys()
            }
            new_genomes.append(genome_child)

        # # Mutations on the newborns
        # print("Mutations on the newborns")
        # mutated_genomes = []
        # for genome in new_genomes:
        #     min_val, max_val = min(genome), max(genome)
        #     mutated_genome = genome.copy()
        #     for i in range(len(genome)):
        #         if np.random.rand() < mutation_rate:
        #             mutated_genome[i] = np.random.uniform(min_val, max_val) # changing the value of the gene
        #     # print(f"replacing old genome \n {genome} by the mutated genome \n {mutated_genome}.")
        #     mutated_genomes.append(mutated_genome)

        # Mutations on the newborns
        print(f"Mutations on the newbords (mutation rate: {mutation_rate})")
        # The mutation is a natural process when you create an new child. Here we do random mutations
        # with the probability of mutation_rate, which means each weight or bias of the neural
        # network has a probability of mutation_rate to be randomly modified. To minimise the
        # complexity, we create a mutation_mask which decides which values will be modified. Then we
        # create an array where all the values have been modified, and finally we update the values
        # of the layer which have been chosen by the mutation_mask to give them the value of the
        # newly created array mutated_values.
        mutated_genomes = []
        for new_genome in new_genomes:
            mutated_genome = {}
            for k in new_genome:
                layer = new_genome[k].copy()
                # print(f"layer.shape {layer.shape}")
                mutation_mask = np.random.rand(len(layer)) < mutation_rate
                mutated_values = np.random.uniform(low=np.min(layer), high=np.max(layer), size=layer.shape)
                # print(f"mutated_values.shape {mutated_values.shape}")
                layer[mutation_mask] = mutated_values[mutation_mask]
                mutated_genome[k] = layer
            mutated_genomes.append(mutated_genome)

        # Printing the best distance
        print(f"The best Bird of Generation {i} got to {max_dist}")
        max_distances.append(max_dist)
        
        # Creating the next generation
        print("Creating the next generation")
        genomes = mutated_genomes

    print(f"Best Distances: {max_distances}")
















if __name__ == "__main__":
    main()