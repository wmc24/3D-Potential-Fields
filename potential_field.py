from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import numpy as np


def get_velocity_to_reach_goal(position, goal_position, dims):
  # MISSING: Compute the velocity field needed to reach goal_position
  # assuming that there are no obstacles.
  
  cruising_speed = 0.5
  convergence_radius = 1
  
  # Getting the distance to the goal
  dist = np.sqrt(np.sum((position - goal_position)**2))
  
  # Getting the direction to the goal
  direction = normalize(goal_position - position)
  
  # Applying a potential inspired by Huber Loss
  if dist < convergence_radius:
    v = cruising_speed * dist * direction / convergence_radius
  else:
    v = cruising_speed * direction

  return v


def get_velocity_to_avoid_planets(position, planet_positions, planet_radii, dims):
  v = np.zeros(dims, dtype=np.float32)
  # MISSING: Compute the velocity field needed to avoid the obstacles
  # In the worst case there might a large force pushing towards the
  # obstacles (consider what is the largest force resulting from the
  # get_velocity_to_reach_goal function). Both obstacle_positions
  # and obstacle_radii are lists.

  return v


def get_velocity_to_avoid_spaceships(position, spaceship_positions, dims):
  v = np.zeros(dims, dtype=np.float32)
  # MISSING: Compute the velocity field needed to avoid other spaceships
  # In the worst case there might a large force pushing towards the
  # obstacles (consider what is the largest force resulting from the
  # get_velocity_to_reach_goal function). 
  
  # spaceship_positions is a list that each element consists of
  # cartesian position coordinates and a normalised direction vector

  return v


def get_velocity_to_avoid_meteoroids(position, meteoroids_positions, dims):
  v = np.zeros(dims, dtype=np.float32)
  # MISSING: Compute the velocity field needed to avoid other spaceships
  # In the worst case there might a large force pushing towards the
  # obstacles (consider what is the largest force resulting from the
  # get_velocity_to_reach_goal function). 
  
  # The meteroid positions are a list consisting of cartesian coordinates and
  # a veclocity vector

  return v


def normalize(v):
  n = np.linalg.norm(v)
  if n < 1e-2:
    return np.zeros_like(v)
  return v / n


def cap(v, max_speed):
  n = np.linalg.norm(v)
  if n > max_speed:
    return v / n * max_speed
  return v


def get_velocity(position, mode='all'):
  if mode in ('goal', 'all'):
    v_goal = get_velocity_to_reach_goal(position, GOAL_POSITION, DIMENSIONS)
  else:
    v_goal = np.zeros(DIMENSIONS, dtype=np.float32)

  if mode in ('planet', 'all'):
    v_planets = get_velocity_to_avoid_planets(position, [PLANET_POSITION], [PLANET_RADIUS], DIMENSIONS)
  else:
    v_planets = np.zeros(DIMENSIONS, dtype=np.float32)
    
  if mode in ('spaceship', 'all'):
    v_spaceships = get_velocity_to_avoid_spaceships(position, [STATIONARY_SPACESHIP], DIMENSIONS)
  else:
    v_spaceships = np.zeros(DIMENSIONS, dtype=np.float32)
    
  if mode in ('meteoroid', 'all'):
    v_meteoroids = get_velocity_to_avoid_meteoroids(position, [METEOROID], DIMENSIONS)
  else:
    v_meteoroids = np.zeros(DIMENSIONS, dtype=np.float32)
    
  v = v_goal + v_planets + v_spaceships + v_meteoroids
  return cap(v, max_speed=MAX_SPEED)


# Default Arena
DIMENSIONS = 2
MAX_SPEED = .5
SIZE_OF_UNIVERSE = 10.
PLANET_POSITION = np.array([3., 2., .5][:DIMENSIONS], dtype=np.float32)
PLANET_RADIUS = .3
STATIONARY_SPACESHIP = np.concatenate((np.array([-.3, 4.], dtype=np.float32),  normalize(np.array([.3, -4.], dtype=np.float32))))
METEOROID = np.concatenate((np.array([-.3, 4.], dtype=np.float32),  MAX_SPEED * 2 * normalize(np.array([.3, -4.], dtype=np.float32))))
GOAL_POSITION = np.array([2.5, 2.5], dtype=np.float32)
START_POSITION = np.array([-2.5, -2.5], dtype=np.float32)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs obstacle avoidance with a potential field')
  parser.add_argument('--mode', action='store', default='all', help='Which velocity field to plot.', choices=['planet', 'spaceship', 'meteoroid', 'goal', 'all'])
  args, unknown = parser.parse_known_args()

  fig, ax = plt.subplots()
  # Plot field.
  X, Y = np.meshgrid(np.linspace(-SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2, 30),
                     np.linspace(-SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2, 30))
  U = np.zeros_like(X)
  V = np.zeros_like(X)
  for i in range(len(X)):
    for j in range(len(X[0])):
      velocity = get_velocity(np.array([X[i, j], Y[i, j]]), args.mode)
      U[i, j] = velocity[0]
      V[i, j] = velocity[1]
  plt.quiver(X, Y, U, V, units='width')

  # Plot environment.
  ax.add_artist(plt.Circle(PLANET_POSITION, PLANET_RADIUS, color='gray'))
  plt.plot([-SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2], [-SIZE_OF_UNIVERSE/2, -SIZE_OF_UNIVERSE/2], 'k')
  plt.plot([-SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2], [SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2], 'k')
  plt.plot([-SIZE_OF_UNIVERSE/2, -SIZE_OF_UNIVERSE/2], [-SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2], 'k')
  plt.plot([SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2], [-SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2], 'k')

  # Plot a simple trajectory from the start position.
  # Uses Euler integration.
  # Also perform Euler integration of the meteoroid
  dt = 0.01
  x = START_POSITION
  x_meteoroid = METEOROID[:DIMENSIONS]
  positions = [x]
  positions_meteoroid = [x_meteoroid]
  for t in np.arange(0., 20., dt):
    v = get_velocity(x, args.mode)
    x = x + v * dt
    positions.append(x)
    v = METEOROID[DIMENSIONS:]
    x_meteoroid = x_meteoroid + v * dt
    positions_meteoroid.append(x_meteoroid)
  positions = np.array(positions)
  positions_meteoroid = np.array(positions_meteoroid)
  plt.plot(positions[:, 0], positions[:, 1], lw=2, c='g')
  plt.plot(positions_meteoroid[:, 0], positions_meteoroid[:, 1], lw=2, c='r')

  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-.5 - SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2 + .5])
  plt.ylim([-.5 - SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2 + .5])
  plt.show()
