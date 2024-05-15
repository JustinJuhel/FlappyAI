import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def threshold_activation(X, threshold):
    return np.where(X > threshold, 1, 0)

class BirdBrain(nn.Module):
    def __init__(self, batch_size, genome=None):
        super().__init__()
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        # Layers
        self.fc1 = nn.Linear(5, 40)
        self.fc2 = nn.Linear(40, 1)

        self.genome = genome
        self.generate_network()
    
    def forward(self, X):
        X = torch.sigmoid(self.fc1(X))
        X = torch.sigmoid(self.fc2(X))
        # y_pred = F.log_softmax(X, dim=1)
        y_pred = threshold_activation(X, threshold=0)
        return y_pred
    
    def generate_network(self): # initializing the weights of the network using the genome of the bird
        if self.genome is None: # if no genome is given, we initialize randomly the weights of the layers
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            # Updating the genome depending on the newly generated weights
            self.genome = np.concatenate((self.fc1.weight.detach().numpy().reshape((-1,)), self.fc2.weight.detach().numpy().reshape((-1,))))
        else: # if a genome is given, we have to initialize the weights of the layers using it
            # Assuming genome is a 1-D array of shape (n_connections,1)
            n_inputs = 5
            n_hidden = 40
            n_outputs = 1

            # Splitting the genome into weights for each layer
            weights1 = self.genome[0][:n_hidden * n_inputs].reshape(n_hidden, n_inputs)
            weights2 = self.genome[0][n_hidden * n_inputs:n_hidden * n_inputs + n_outputs * n_hidden].reshape(n_outputs, n_hidden)

            # Assigning the weights to the layers
            self.fc1.weight.data = torch.from_numpy(weights1)
            self.fc2.weight.data = torch.from_numpy(weights2)

            
