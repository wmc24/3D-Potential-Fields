import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicGoalNet(nn.Module):
    def __init__(self, dims):
      super(BasicGoalNet, self).__init__()
      # Temporary model that has been designed for the goal potential only
      # so only inputs three variables which are the relative coordinates
      # of the goal to the spaceship - it is simple so it is easy to train
      self.linear1 = nn.Linear(dims, 16)
      self.linear2 = nn.Linear(16, dims)

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
    def __init__(self, dims):
      super(Net, self).__init__()
      # Temporary model that has been designed for the goal potential only
      # so only inputs three variables which are the relative coordinates
      # of the goal to the spaceship - it is simple so it is easy to train
      self.goal_linear1 = nn.Linear(dims, 16)
      self.goal_linear2 = nn.Linear(16, dims)
      self.planet_linear1 = nn.Linear(dims+1, 16)
      self.planet_linear2 = nn.Linear(16, dims)

    # The passed arguements are of the dimensions:
    # goal_position - (dims)
    # planet_positions - (N, dims)
    # planet_radii - (N, 1)
    # spaceships - (N, dims*2)
    # meteoroids - (N, dims*2)
    def forward(self, goal_position, planets_position, planets_radii, spaceships, meteoroids):
      # Computing forward passes for a general scenario input to make the network friendlier
      # to work with elsewhere
      if goal_position.size()[1] != 0:
        print(goal_position.size())
        goal_output = self.forward_goal(goal_position)
      else:
        goal_output = torch.zeros(dims)

      planets_output = torch.zeros(dims)
      if planets_position.size()[1] != 0:
        for i in range(planets_position.size()[0]):
          planets_output += self.forward_planet(planets_position[i, :], planets_radii[i, :])
          
      # For now for spaceships and meteoroids we use the planets 
      spaceships_output = torch.zeros(dims)
      if spaceships.size()[1] != 0:
        for i in range(spaceships.size()[0]):
          spaceships_output += self.forward_spaceship(spaceships[i, :])

      meteoroids_output = torch.zeros(dims)
      if meteoroids.size()[1] != 0:
        for i in range(meteoroids.size()[0]):
          meteoroids_output += self.forward_meteoroid(meteoroids[i, :])

      # Combining these together into a single output
      output = goal_output + planets_output + spaceships_output + meteoroids_output

      # Returning multiple outputs for tensorboard plotting
      return output, goal_output, planets_output, spaceships_output, meteoroids_output
    
    def forward_goal(self, goal_position):
      # Computing the forward pass for a goal potential only
      # Pass data through first linear layer
      x = self.goal_linear1(goal_position)
      # Use leaky ReLU
      x = F.leaky_relu(x)
      # Pass data through second linear layer
      x = self.goal_linear2(x)
      output = torch.tanh(x)
      return output

    def forward_planet(self, planet_position, planet_radius):
      # Computing the forward pass for a single planet potential only
      x = torch.cat(planet_position, planet_radius)
      # Pass data through first linear layer
      x = self.planet_linear1(x)
      # Use leaky ReLU
      x = F.leaky_relu(x)
      # Pass data through second linear layer
      x = self.planet_linear2(x)
      output = torch.tanh(x)
      return output
    
    def forward_spaceship(self, spaceship):
      # Temporarily using the same as the planets net
      return self.forward_planet(spaceship[:-1], spaceship[-1])

    def forward_meteoroid(self, meteoriod):
      # Temporarily using the same as the planets net
      return self.forward_planet(meteoriod[:-1], meteoriod[-1])
