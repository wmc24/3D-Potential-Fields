import torch
import torch.nn as nn
import torch.nn.functional as F

def whitening(x, device):
  # Input is a tensor of (N, variable) dims
  # returns its mean and standard deviation
  means = torch.mean(x).float().to(device)
  sigma = torch.std(x).float().to(device)
  return means, sigma


class BasicGoalNet(nn.Module):
    def __init__(self, dims):
      super(BasicGoalNet, self).__init__()
      # Temporary model that has been designed for the goal potential only
      # so only inputs three variables which are the relative coordinates
      # of the goal to the spaceship - it is simple so it is easy to train
      self.linear1 = nn.Linear(dims, 16)
      self.linear2 = nn.Linear(16, dims)
      self.whitening = None

    # The passed arguements are of the dimensions:
    # goal_disp - (dims)
    # planet_disps - (N, dims)
    # planet_radii - (N, 1)
    # spaceships - (N, dims*2)
    # meteoroids - (N, dims*2)
    def forward(self, goal_disp, planets_dist, planets_radii, spaceships, meteoroids):
      pass
    
    def forward_goal(self, goal_disp):
      # Pass data through first linear layer
      x = self.linear1(goal_disp)
      # Use leaky ReLU
      x = F.leaky_relu(x)
      # Pass data through second linear layer
      x = self.linear2(x)
      # Use tanh actiavtion function as we scale
      # the output by the max speed - NOT IMPLEMENTED
      output = torch.tanh(x)

      return output

    def forward_obstacle(self, dist, size):
      return 0
    
    def switch_nonlinearity(self):
      pass

    def data_whitening(self, goal_disp, planets_dist, planets_radii, spaceships_dist, spaceships_size, meteoroid_dist, meteoroid_size, device):
      pass


class Net(nn.Module):
    def __init__(self, dims=3):
      super(Net, self).__init__()
      self.dims = dims
      self.leaky = True
      self.num_neurons1 = 16
      self.num_neurons2 = 8

      self.goal_linear1 = nn.Linear(1, self.num_neurons1)
      self.goal_linear2 = nn.Linear(self.num_neurons1, self.num_neurons1)
      self.goal_linear3 = nn.Linear(self.num_neurons1, self.num_neurons2)
      self.goal_linear4 = nn.Linear(self.num_neurons2, 1)

      self.obstacle_linear1 = nn.Linear(2, self.num_neurons1)
      self.obstacle_linear2 = nn.Linear(self.num_neurons1, self.num_neurons1)
      self.obstacle_linear3 = nn.Linear(self.num_neurons1, self.num_neurons2)
      self.obstacle_linear4 = nn.Linear(self.num_neurons2, self.num_neurons2)
      self.obstacle_linear5 = nn.Linear(self.num_neurons2, 1)

      self.moving_obstacle_linear1 = nn.Linear(4, self.num_neurons1)
      self.moving_obstacle_linear2 = nn.Linear(self.num_neurons1, self.num_neurons1)
      self.moving_obstacle_linear3 = nn.Linear(self.num_neurons1, self.num_neurons1)
      self.moving_obstacle_linear4 = nn.Linear(self.num_neurons1, self.num_neurons2)
      self.moving_obstacle_linear5 = nn.Linear(self.num_neurons2, self.num_neurons2)
      self.moving_obstacle_linear6 = nn.Linear(self.num_neurons2, self.num_neurons2)
      self.moving_obstacle_linear7 = nn.Linear(self.num_neurons2, 1)

      self.whitening_goal_mean = 0
      self.whitening_goal_sigma = 1
      self.whitening_obstacle_mean = 0
      self.whitening_obstacle_sigma = 1
      self.whitening_obstacle_size_mean = 0
      self.whitening_obstacle_size_sigma = 1
      self.whitening_moving_obstacle_mean = 0
      self.whitening_moving_obstacle_sigma = 1
      self.whitening_moving_obstacle_size_mean = 0
      self.whitening_moving_obstacle_size_sigma = 1
      self.whitening_moving_obstacle_speed_mean = 0
      self.whitening_moving_obstacle_speed_sigma = 1

      self.whitening = (self.whitening_goal_mean,
                        self.whitening_goal_sigma,
                        self.whitening_obstacle_mean,
                        self.whitening_obstacle_sigma,
                        self.whitening_obstacle_size_mean,
                        self.whitening_obstacle_size_sigma,
                        self.whitening_moving_obstacle_mean,
                        self.whitening_moving_obstacle_sigma,
                        self.whitening_moving_obstacle_size_mean,
                        self.whitening_moving_obstacle_size_sigma,
                        self.whitening_moving_obstacle_speed_mean,
                        self.whitening_moving_obstacle_speed_sigma)

    def data_whitening(self, goal_disp, planets_dist, planets_radii, spaceships_dist, spaceships_size, meteoroids_disp, meteoroids_size, meteoroids_speed, device):
      # Computing a data whitening matrix for all the nets to improve training at beginning
      self.whitening_goal_mean, self.whitening_goal_sigma = whitening(torch.norm(goal_disp, dim=1, p=2).reshape((-1, 1)), device)

      obstacle_dists = torch.cat((planets_dist, spaceships_dist))
      self.whitening_obstacle_mean, self.whitening_obstacle_sigma = whitening(obstacle_dists, device)
      obstacle_sizes = torch.cat((planets_radii, spaceships_size))
      self.whitening_obstacle_size_mean, self.whitening_obstacle_size_sigma = whitening(obstacle_sizes, device)

      self.whitening_moving_obstacle_mean, self.whitening_moving_obstacle_sigma = whitening(meteoroids_disp, device)
      self.whitening_moving_obstacle_size_mean, self.whitening_moving_obstacle_size_sigma = whitening(meteoroids_size, device)
      self.whitening_moving_obstacle_speed_mean, self.whitening_moving_obstacle_speed_sigma = whitening(meteoroids_speed, device)

      self.whitening = (self.whitening_goal_mean,
                        self.whitening_goal_sigma,
                        self.whitening_obstacle_mean,
                        self.whitening_obstacle_sigma,
                        self.whitening_obstacle_size_mean,
                        self.whitening_obstacle_size_sigma,
                        self.whitening_moving_obstacle_mean,
                        self.whitening_moving_obstacle_sigma,
                        self.whitening_moving_obstacle_size_mean,
                        self.whitening_moving_obstacle_size_sigma,
                        self.whitening_moving_obstacle_speed_mean,
                        self.whitening_moving_obstacle_speed_sigma)
    
    def forward_goal(self, goal_disp):
      # Computing the forward pass for a goal potential only.
      # It only depends upon the dispance so we get a L2 norm
      # and scale the direction by the output of the network
      # Pass data through first linear layer

      dist = torch.norm(goal_disp, dim=1, p=2).reshape((-1, 1))
      dist = torch.where(dist == 0.0, torch.tensor(1.0).to(dist.device), dist)
      dist_cat = torch.cat((dist, dist), dim=1)
      if goal_disp.size()[1] == 3:
        dist_cat = torch.cat((dist_cat, dist), dim=1)
      direction = -torch.div(goal_disp, dist_cat)

      # Performing data whitening
      dist = (dist - self.whitening[0]) / self.whitening[1]

      x = self.goal_linear1(dist)
      x = F.leaky_relu(x)
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

    def forward_obstacle(self, dist, size):
      # Computing the forward pass for a single object potential only.
      # It only depends upon the distance and radius so we get a L2 norm
      # and scale the direction by the output of the network

      # Performing data whitening
      dist = (dist - self.whitening[2]) / self.whitening[3]
      size = (size - self.whitening[4]) / self.whitening[5]
      
      #x = torch.cat((dist.reshape((-1, 1)) - size.reshape((-1, 1)), size.reshape((-1, 1))), dim=1)
      x = torch.cat((dist.reshape((-1, 1)), size.reshape((-1, 1))), dim=1)
      x = self.obstacle_linear1(x)
      x = F.leaky_relu(x)
      x = self.obstacle_linear2(x)
      x = F.leaky_relu(x)
      x = self.obstacle_linear3(x)
      x = F.leaky_relu(x)
      x = self.obstacle_linear4(x)
      x = F.leaky_relu(x)
      x = self.obstacle_linear5(x)
      if self.leaky is True and self.training is True:
        x = F.leaky_relu(x)
      else:
        x = F.relu(x)

      return x

    def forward_moving_obstacle(self, disp, size, speed):
      # Computing the forward pass for a single moving object potential only.
      # It only depends upon the distance and radius so we get a L2 norm
      # and scale the direction by the output of the network

      # Performing data whitening
      disp = (disp - self.whitening[6]) / self.whitening[7]
      size = (size - self.whitening[8]) / self.whitening[9]
      speed = (size - self.whitening[10]) / self.whitening[11]
      
      #x = torch.cat((dist.reshape((-1, 1)) - size.reshape((-1, 1)), size.reshape((-1, 1))), dim=1)
      x = torch.cat((disp.reshape((-1, 2)), size.reshape((-1, 1)), speed.reshape((-1, 1))), dim=1)
      x = self.moving_obstacle_linear1(x)
      x = F.leaky_relu(x)
      x = self.moving_obstacle_linear2(x)
      x = F.leaky_relu(x)
      x = self.moving_obstacle_linear3(x)
      x = F.leaky_relu(x)
      x = self.moving_obstacle_linear4(x)
      x = F.leaky_relu(x)
      x = self.moving_obstacle_linear5(x)
      x = F.leaky_relu(x)
      x = self.moving_obstacle_linear6(x)
      x = F.leaky_relu(x)
      x = self.moving_obstacle_linear7(x)
      x = torch.tanh(x)

      return x
    
    def switch_nonlinearity(self):
      # if enough training has occured we switch the non-linearity
      # from leaky RelU to ReLU
      self.leaky = False
