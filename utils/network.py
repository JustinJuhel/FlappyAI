import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

def threshold_activation(X, threshold):
    return np.where(X > threshold, 1, 0)

class BirdBrain(nn.Module):
    def __init__(self, batch_size, genome=None):
        super().__init__()
        # self.batch_size = batch_size
        self.batch_size = 1
        self.criterion = nn.CrossEntropyLoss()
        # Layers
        self.fc1 = nn.Linear(4, 40)
        self.fc2 = nn.Linear(40, 1)
        self.network = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.genome = genome  # This is a Python Dictionary contianing all the weights and bias of all layers.
        self.generate_network()
    
    def forward(self, X):
        return self.network(X)
    
    '''
    This Function generates a neural network for the bird's brain, which is supposed to control the bird.
    Depending on whether or not the genome is None, we instantiate the model with random weights of
    Chose weights, corresponding to the output genome of the reproduction between two reproducer birds.
    '''

    def generate_network(self):
        if self.genome is None:  # If it's the first generation, genome is None.
            # The Fully Connected Layers are getting randomly initialised
            for m in self.network.modules():
                if isinstance(m, nn.Linear):
                    init.uniform_(m.weight, a=-0.1, b=0.1)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
            # The genome takes the same values
            weights_dict = {k: v.detach().numpy() for k, v in self.network.named_parameters()}
            self.genome = weights_dict
        else:  # If the Genome is not None, we have to adapt the weights of the network with the given genome
            tensor_genome = {
                k: torch.Tensor(self.genome[k]) for k in self.genome.keys()
            }
            self.network.load_state_dict(tensor_genome)
            """S'il y a une erreur là il faut adapter les noms du dictionnaire qu'on veut load dans le NN parce que ça peut ne pas correspondre"""

    '''
    This Function takes a (4,)-shaped array in input and returns the response of the bird's brain,which is juste one numerical value
    '''

    def get_response(self, input, next_pipe_dist, next_pipe_height):
        if len(input.shape) == 1:
            input_extended = np.concatenate((
                input,
                np.reshape(np.array(next_pipe_dist), (1,)),  # Getting a (1,)-shaped array from an integer.
                np.reshape(np.array(next_pipe_height), (1,)),  # Getting a (1,)-shaped array from an integer.
            ))
        else:
            col_ones = np.ones_like(input[:, 0:1])  # We create a column array full of 1 of the same length as the columns of the input
            input_extended = np.concatenate((input, col_ones * next_pipe_dist, col_ones * next_pipe_height), axis=1)
        t_input = torch.Tensor(input_extended)
        return self(t_input)