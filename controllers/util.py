import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, no_of_inputs = 115, no_of_outputs = 19):
        self.no_of_layers = 2
        self.no_of_hidden_units = 100
        super(DQN, self).__init__()

        self.lin_trans = []

        self.lin_trans.append(nn.Linear(no_of_inputs, self.no_of_hidden_units))
        self.lin_trans.append(torch.nn.ReLU())
        for layer_index in range(self.no_of_layers-2):
            self.lin_trans.append(nn.Linear(self.no_of_hidden_units, self.no_of_hidden_units))
            self.lin_trans.append(torch.nn.ReLU())

        self.lin_trans.append(nn.Linear(self.no_of_hidden_units, no_of_outputs))
        self.lin_trans.append(torch.nn.Sigmoid())
        self.lin_trans.append(torch.nn.Softmax())

        self.model = torch.nn.Sequential(*self.lin_trans)

    def forward(self, x):
        y = self.model(x)
        return y

from collections import namedtuple, deque
import random 

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

