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
            nn.Linear(4, 120),
            nn.ReLU(),
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.Linear(120, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.genome = genome  # This is a Python Dictionary contianing all the weights and bias of all layers.
        self.generate_network()
    
    # def forward(self, X):
    #     X = torch.sigmoid(self.fc1(X))
    #     X = torch.sigmoid(self.fc2(X))
    #     # y_pred = F.log_softmax(X, dim=1)
    #     y_pred = threshold_activation(X, threshold=0)
    #     return y_pred

    def forward(self, X):
        return self.network(X)
    
    # def generate_network(self): # initializing the weights of the network using the genome of the bird
    #     if self.genome is None: # if no genome is given, we initialize randomly the weights of the layers
    #         nn.init.xavier_uniform_(self.fc1.weight)
    #         nn.init.xavier_uniform_(self.fc2.weight)
    #         # Updating the genome depending on the newly generated weights
    #         self.genome = np.concatenate((self.fc1.weight.detach().numpy().reshape((-1,)), self.fc2.weight.detach().numpy().reshape((-1,))))
    #     else: # if a genome is given, we have to initialize the weights of the layers using it
    #         # Assuming genome is a 1-D array of shape (n_connections,1)
    #         n_inputs = 4
    #         n_hidden = 40
    #         n_outputs = 1

    #         # Splitting the genome into weights for each layer
    #         weights1 = self.genome[:n_hidden * n_inputs].reshape(n_hidden, n_inputs)
    #         weights2 = self.genome[n_hidden * n_inputs:n_hidden * n_inputs + n_outputs * n_hidden].reshape(n_outputs, n_hidden)

    #         # Assigning the weights to the layers
    #         self.fc1.weight.data = torch.from_numpy(weights1)
    #         self.fc2.weight.data = torch.from_numpy(weights2)

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

    # This Function takes a (4,)-shaped array in input and returns the response of the bird's brain, which is juste one numerical value

    def get_response(self, input):
        # print(f"get_response -> input shape: {input.shape}")
        t_input = torch.Tensor(input)
        # print(f"get_respone -> Tensor Input: {t_input}, shape {t_input.shape}")
        output = self(t_input)
        # print(f"get_response -> output: {output}, type {type(output)}")
        return output