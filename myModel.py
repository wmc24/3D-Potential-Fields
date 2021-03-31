import torch
import torch.nn as nn
import torch.nn.functional as F

def whitening(x, device):
  # Input is a tensor of (N, variable) dims
  # returns its mean and standard deviation over the
  # N dimension
  means = torch.mean(x, dim=0).float().to(device)
  sigma = torch.std(x, dim=0).float().to(device)
  return means, sigma


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
      self.dims = dims
      self.leaky = True
      self.goal_linear1 = nn.Linear(1, 16)
      self.goal_linear2 = nn.Linear(16, 8)
      self.goal_linear3 = nn.Linear(8, 8)
      self.goal_linear4 = nn.Linear(8, 1)

      self.planet_linear1 = nn.Linear(2, 16)
      self.planet_linear2 = nn.Linear(16, 16)
      self.planet_linear3 = nn.Linear(16, 8)
      self.planet_linear4 = nn.Linear(8, 8)
      self.planet_linear5 = nn.Linear(8, 1)

      self.spaceship_linear1 = nn.Linear(2, 16)
      self.spaceship_linear2 = nn.Linear(16, 16)
      self.spaceship_linear3 = nn.Linear(16, 8)
      self.spaceship_linear4 = nn.Linear(8, 8)
      self.spaceship_linear5 = nn.Linear(8, 1)

      self.meteoroid_linear1 = nn.Linear(2, 16)
      self.meteoroid_linear2 = nn.Linear(16, 16)
      self.meteoroid_linear3 = nn.Linear(16, 8)
      self.meteoroid_linear4 = nn.Linear(8, 8)
      self.meteoroid_linear5 = nn.Linear(8, 1)

    def data_whitening(self, goal_position, planets_position, planets_radii, spaceships, meteoroids, device):
      # Computing a data whitening matrix for all the nets to improve training at beginning
      self.whitening_goal_mean, self.whitening_goal_sigma = whitening(goal_position, device)
      self.whitening_planet_position_mean, self.whitening_planet_position_sigma = whitening(planets_position, device)
      self.whitening_planet_radii_mean, self.whitening_planet_radii_sigma = whitening(planets_radii, device)
      self.whitening_spaceships_mean, self.whitening_spaceships_sigma = whitening(spaceships, device)
      self.whitening_meteoroids_mean, self.whitening_meteoroids_sigma = whitening(meteoroids, device)

    # The passed arguements are of the dimensions:
    # goal_position - (dims)
    # planet_positions - (N, dims)
    # planet_radii - (N, 1)
    # spaceships - (N, dims*2)
    # meteoroids - (N, dims*2)
    def forward(self, goal_position, planets_position, planets_radii, spaceships, meteoroids):
      # Computing forward passes for a general scenario input to make the network friendlier
      # to work with elsewhere

      """
      # Performing data whitening
      goal_position = (goal_position - self.whitening_goal_mean) / self.whitening_goal_sigma
      planets_position = (planets_position - self.whitening_planet_position_mean) / self.whitening_planet_position_sigma
      planets_radii = (planets_radii - self.whitening_planet_radii_mean) / self.whitening_planet_radii_sigma
      spaceships = (spaceships - self.whitening_spaceships_mean) / self.whitening_spaceships_sigma
      meteoroids = (meteoroids - self.whitening_meteoroids_mean) / self.whitening_meteoroids_sigma
      """
      batch_size = goal_position.size()[0]
      device = goal_position.get_device()

      if goal_position.size()[1] != 0:
        goal_output = self.forward_goal(goal_position.squeeze())
      else:
        goal_output = torch.zeros(self.dims)

      planets_output = torch.zeros(batch_size, self.dims).to(device)
      if planets_position.size()[1] != 0:
        for i in range(planets_position.size()[1]):
          planets_output += self.forward_planet(planets_position[:, i, :].squeeze(), planets_radii[:, i, :].squeeze())
          
      # For now for spaceships and meteoroids we use the planets 
      spaceships_output = torch.zeros(batch_size, self.dims).to(device)
      if spaceships.size()[1] != 0:
        for i in range(spaceships.size()[1]):
          spaceships_output += self.forward_spaceship(spaceships[:, i, :].squeeze())

      meteoroids_output = torch.zeros(batch_size, self.dims).to(device)
      if meteoroids.size()[1] != 0:
        for i in range(meteoroids.size()[1]):
          meteoroids_output += self.forward_meteoroid(meteoroids[:, i, :].squeeze())

      # Combining these together into a single output
      output = goal_output + planets_output + spaceships_output + meteoroids_output

      # Returning multiple outputs for tensorboard plotting
      return output, goal_output, planets_output, spaceships_output, meteoroids_output
    
    def forward_goal(self, goal_position):
      # Computing the forward pass for a goal potential only
      # It only depends upon the distance so we get a L2 norm
      # and scale the direction by the output of the network
      # Pass data through first linear layer

      # Performing data whitening
      goal_position = (goal_position - self.whitening_goal_mean) / self.whitening_goal_sigma

      mag = torch.norm(goal_position, p=2, dim=1).reshape((-1, 1))
      direction = goal_position / mag

      x = self.goal_linear1(mag)
      x = torch.sigmoid(x)
      x = self.goal_linear2(x)
      x = F.leaky_relu(x)
      x = self.goal_linear3(x)
      x = F.leaky_relu(x)
      x = self.goal_linear4(x)
      if self.leaky is True and self.training is True:
        x = F.leaky_relu(x)
      else:
        x = F.relu(x)

      return direction * x

    def forward_planet(self, planet_position, planet_radius):
      # Computing the forward pass for a single planet potential only
      # It only depends upon the distance and radius so we get a L2 norm
      # and scale the direction by the output of the network

      # Performing data whitening
      planet_position = (planet_position - self.whitening_planet_position_mean) / self.whitening_planet_position_sigma
      planet_radius = (planet_radius - self.whitening_planet_radii_mean) / self.whitening_planet_radii_sigma

      mag = torch.norm(planet_position, p=2, dim=1).reshape((-1, 1))
      direction = - planet_position / mag # Negative as we want to avoid planets
      
      x = torch.cat((mag, planet_radius.reshape((-1, 1))), dim=1)
      x = self.planet_linear1(x)
      x = torch.sigmoid(x)
      x = self.planet_linear2(x)
      x = torch.sigmoid(x)
      x = self.planet_linear3(x)
      x = F.leaky_relu(x)
      x = self.planet_linear4(x)
      x = F.leaky_relu(x)
      x = self.planet_linear5(x)
      if self.leaky is True and self.training is True:
        x = F.leaky_relu(x)
      else:
        x = F.relu(x)

      return direction * x
    
    def forward_spaceship(self, spaceship):
      # Computing the forward pass for a single planet spaceship only
      # It only depends upon the distance and radius so we get a L2 norm
      # and scale the direction by the output of the network

      # Performing data whitening
      spaceship = (spaceship - self.whitening_spaceships_mean) / self.whitening_spaceships_sigma

      mag = torch.norm(spaceship[:, :-1], p=2, dim=1).reshape((-1, 1))
      direction = - spaceship[:, :-1] / mag # Negative as we want to avoid spaceships
      
      x = torch.cat((mag, spaceship[:, -1].reshape((-1, 1))), dim=1)
      x = self.spaceship_linear1(x)
      x = torch.sigmoid(x)
      x = self.spaceship_linear2(x)
      x = torch.sigmoid(x)
      x = self.spaceship_linear3(x)
      x = F.leaky_relu(x)
      x = self.spaceship_linear4(x)
      x = F.leaky_relu(x)
      x = self.spaceship_linear5(x)
      if self.leaky is True and self.training is True:
        x = F.leaky_relu(x)
      else:
        x = F.relu(x)

      return direction * x

    def forward_meteoroid(self, meteoroid):
      # Computing the forward pass for a single planet spaceship only
      # It only depends upon the distance and radius so we get a L2 norm
      # and scale the direction by the output of the network

      # Performing data whitening
      meteoroid = (meteoroid - self.whitening_meteoroids_mean) / self.whitening_meteoroids_sigma

      mag = torch.norm(meteoroid[:, :-1], p=2, dim=1).reshape((-1, 1))
      direction = - meteoroid[:, :-1] / mag # Negative as we want to avoid meteoroids
      
      x = torch.cat((mag, meteoroid[:, -1].reshape((-1, 1))), dim=1)
      x = self.meteoroid_linear1(x)
      x = torch.sigmoid(x)
      x = self.meteoroid_linear2(x)
      x = torch.sigmoid(x)
      x = self.meteoroid_linear3(x)
      x = F.leaky_relu(x)
      x = self.meteoroid_linear4(x)
      x = F.leaky_relu(x)
      x = self.meteoroid_linear5(x)
      if self.leaky is True and self.training is True:
        x = F.leaky_relu(x)
      else:
        x = F.relu(x)
        
      return direction * x
    
    def switch_nonlinearity(self):
      # if enough training has occured we switch the non-linearity
      # from leaky RelU to ReLU
      self.leaky = False
