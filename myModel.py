import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      # Temporary model that has been designed for the goal potential only
      # so only inputs three variables which are the relative coordinates
      # of the goal to the spaceship - it is simple so it is easy to train
      self.linear1 = nn.Linear(3, 16)
      self.linear2 = nn.Linear(16, 3)

    # x represents our data
    def forward(self, x):
      # Pass data through first linear layer
      x = self.linear1(x)
      # Use leaky ReLU
      x = F.leaky_relu(x)
      # Pass data through second linear layer
      x = self.linear1(x)
      # Use tanh actiavtion function as we scale
      # the output by the max speed
      output = F.tanh(x)

      return output