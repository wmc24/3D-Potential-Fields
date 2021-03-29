import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicGoalNet(nn.Module):
    def __init__(self):
      super(BasicGoalNet, self).__init__()
      # Temporary model that has been designed for the goal potential only
      # so only inputs three variables which are the relative coordinates
      # of the goal to the spaceship - it is simple so it is easy to train
      self.linear1 = nn.Linear(3, 16)
      self.linear2 = nn.Linear(16, 3)

    # The passed arguements are of the dimensions:
    # goal_position - (dims)
    # planet_positions - (N, dims)
    # planet_radii - (N, 1)
    # spaceships - (N, dims*2)
    # meteoroids - (N, dims*2)
    def forward(self, goal_position, planets_position, planets_radii, spaceships, meteoroids):
      # Pass data through first linear layer
      x = self.linear1(goal_position)
      # Use leaky ReLU
      x = F.leaky_relu(x)
      # Pass data through second linear layer
      x = self.linear2(x)
      # Use tanh actiavtion function as we scale
      # the output by the max speed - NOT IMPLEMENTED
      output = torch.tanh(x)

      goal_output = output

      return output, goal_output


class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      # Temporary model that has been designed for the goal potential only
      # so only inputs three variables which are the relative coordinates
      # of the goal to the spaceship - it is simple so it is easy to train
      self.goal_linear1 = nn.Linear(3, 16)
      self.goal_linear2 = nn.Linear(16, 3)

    # The passed arguements are of the dimensions:
    # goal_position - (dims)
    # planet_positions - (N, dims)
    # planet_radii - (N, 1)
    # spaceships - (N, dims*2)
    # meteoroids - (N, dims*2)
    def forward(self, goal_position, planets_position, planets_radii, spaceships, meteoroids):
      # Computing forward passes for each respective net if a position/obstacle etc is passed
      if goal_position.size()[0] != 0:
        goal_output = self.forward_goal(goal_position)
      else:
        goal_output = torch.zeros(3)

      planets_output = torch.zeros(3)
      if planets_position.size()[0] != 0:
        for i in range(planets_position.size()[0])
          planets_output += self.forward_planets(planets_position[i, :], planets_radii[i, :])
          
      # For now for spaceships and meteoroids we use the planets 
      spaceships_output = torch.zeros(3)
      if spaceships.size()[0] != 0:
        for i in range(spaceships.size()[0])
          spaceships_output += self.forward_planets(spaceships[i, :])

      meteoroids_output = torch.zeros(3)
      if meteoroids.size()[0] != 0:
        for i in range(meteoroids.size()[0])
          meteoroids_output += self.forward_planets(meteoroids[i, :])

      # Combining these together into a single output
      output = goal_output + planets_output + spaceships_output + meteoroids_output

      # Returning multiple outputs for tensorboard plotting
      return output, goal_output, planets_output, spaceships_output, meteoroids_output
    
    def forward_goal(self, goal_position):
      # Pass data through first linear layer
      x = self.goal_linear1(goal_position)
      # Use leaky ReLU
      x = F.leaky_relu(x)
      # Pass data through second linear layer
      x = self.goal_linear2(x)
      output = x
      return output

    def forward_planets(self, obstacle_position, obstacle_radius):
      # MISSING
      return output