import torch
import torch.nn as nn

class BrokenHopfieldNetwork:
   def __init__(self, input, subnet_size, activation_threshold):
       self.subnet_size = subnet_size
       self.activation_threshold = activation_threshold
       assert input_shape[0] > input_shape[3]#
   #def train

class HopfieldNetwork:
    def __init__(self, input_shape, activation_threshold):
        self.activation_threshold = activation_threshold
        assert input_shape[0] > input_shape[2]

        weights = torch.zeros(input_shape[0], input_shape[0]).to('cuda')
        nodes = torch.zeros(input_shape).to('cuda')

    def weight_init(self, input):
        self.weights = torch.outer(input.flatten(), input.flatten())
        
        diagonal = torch.diag_embed(torch.diag(self.weights))
        self.weights = self.weights - diagonal
        
    def run(self, input, limit, asyng = False):
        last_energy = self.energy(input)
        self.nodes = self.weights * input.flatten()

        for _ in range(limit - 1):
            if last_energy == self.energy(self.nodes):
                print("Converged!")
                return
            
            self.nodes = self.weights * input.flatten()
            last_energy = self.energy(self.nodes)
            
    def energy(self, nodes):
        return -0.5 * (torch.sum(self.weights * nodes.flatten())) + torch.sum(nodes * self.activation_threshold)