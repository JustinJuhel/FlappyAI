# FlappyAI
  
## Introduction
Welcome to FlappyAI, a simple project that takes inspiration from the popular game, Flappy Bird, to develop and train an unsupervised learning algorithm.
  
Flappy Bird, for those who might not be familiar, is a popular mobile game that features a small bird navigating through a series of pipes. The bird moves forward continuously, and the player's task is to tap the screen to make the bird flap its wings and ascend. The challenge lies in timing the taps correctly to avoid hitting the pipes.
  
In this project, we're not just playing Flappy Bird - we're teaching an AI model how to play it. The goal is to create an AI that can learn and improve its Flappy Bird skills. To achieve this, we're using a genetic algorithm, a method that works by mimicking the process of natural evolution.
## Genetic algorithm
The Genetic Algorithm is a search heuristic that is inspired by the process of natural selection. The algorithm simulates the evolution of a population of solutions to find the best solution.
  
Here's a simplified explanation of how the Genetic Algorithm works:
1. **Initialization**: Start with a population of candidate solutions, which are often represented as chromosomes or strings of genes. Each gene can take on a specific value or be a binary digit (0 or 1). In our case, we have a population composed of numerous birds, and each of them is equipped with a brain (a simple neural network). Initially the networks' weights are randomly chosen.
2. **Fitness Evaluation**: Evaluate the fitness of each candidate solution based on a fitness function. The fitness function measures how well each solution solves the problem at hand. For our FlappyAI game, we measure the performance of a bird by the distance it was able to run before dying against a frontier (sky, ground or pipe).
3. **Selection**: Select a predefined number parent solutions from the population based on their fitness. Solutions with higher fitness (the birds which play Flappy Bird the best) are more likely to be selected.
4. **Crossover**: Create offspring by combining the genes of two parent solutions for each component of the new population. This is done through a process called crossover, which involves exchanging genes between the parents to create new solutions. It is exactly as if we were taking the 10% best birds and make them reproduce two by two to generate enough new birds for the next generation's population.
5. **Mutation**: Introduce random changes to the offspring's genes to maintain diversity in the population. This is done through a process called mutation, which helps the algorithm explore new areas of the solution space.
6. **Replacement**: Replace the least fit solutions in the population with the offspring created through crossover and mutation.
7. **Iteration**: Repeat steps 3 to 6 for a specified number of generations or until a termination condition is met.
8. **Termination**: Stop the algorithm when a satisfactory solution is found or when a maximum number of generations has been reached.
  
The Genetic Algorithm continues to evolve the population of solutions over time, with each generation becoming better than the previous one. The algorithm's goal is to find the best solution, which is typically the one with the highest fitness value.
## Object-Oriented Programming (OOP)
To make my code clearer I used Object-Oriented Programming and created several classes like `Population`, `Pipe`, and `BirdBrain` (inheriting from `nn.Module`).
  
Object-Oriented Programming (OOP) in Python allows you to create classes, which are blueprints for creating objects. Objects are instances of classes and can have attributes (data) and methods (functions) associated with them. OOP in Python provides a way to structure and organize code, making it easier to manage and maintain large codebases. It also allows for the creation of reusable and extensible components, which can be used in a variety of applications. Key concepts include classes, objects, attributes, methods, encapsulation, inheritance, and polymorphism.