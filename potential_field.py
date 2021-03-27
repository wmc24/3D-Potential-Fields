from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import numpy as np


def get_velocity_to_reach_goal(position, goal_position, dims, cruising_speed = 0.5, convergence_radius = 1):
  # MISSING: Compute the velocity field needed to reach goal_position
  # assuming that there are no obstacles.

  # positions and goal_positions are numpy arrays for a single instance of a spaceship and its goal

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
  # get_velocity_to_reach_goal function). Both obstacle_positions (N, dims)
  # and obstacle_radii are numpy arrays (N, 1), the first dimension correponds
  # each planet.

  return v


def get_velocity_to_avoid_spaceships(position, spaceship_positions, dims):
  v = np.zeros(dims, dtype=np.float32)
  # MISSING: Compute the velocity field needed to avoid other spaceships
  # In the worst case there might a large force pushing towards the
  # obstacles (consider what is the largest force resulting from the
  # get_velocity_to_reach_goal function). 
  
  # spaceship_positions is a numpy array, the first dimension correponds
  # each spaceship (N, dims*2). that each element consists of
  # cartesian position coordinates and a normalised direction vector

  return v


def get_velocity_to_avoid_meteoroids(position, meteoroids, dims):
  v = np.zeros(dims, dtype=np.float32)
  # MISSING: Compute the velocity field needed to avoid other spaceships
  # In the worst case there might a large force pushing towards the
  # obstacles (consider what is the largest force resulting from the
  # get_velocity_to_reach_goal function). 
  
  # meteroids is a numpy array (N, dims*2), the first dimension correponds
  # each meteoroid. consisting of cartesian coordinates and
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


def get_velocity(position, goal_position, planet_positions, planet_radii, other_spaceships, meteoroids, dims, max_speed, mode='all'):
  if mode in ('goal', 'all'):
    v_goal = get_velocity_to_reach_goal(position, goal_position, dims)
  else:
    v_goal = np.zeros(dims, dtype=np.float32)

  if mode in ('planet', 'all'):
    v_planets = get_velocity_to_avoid_planets(position, planet_positions, planet_radii, dims)
  else:
    v_planets = np.zeros(dims, dtype=np.float32)
    
  if mode in ('spaceship', 'all'):
    v_spaceships = get_velocity_to_avoid_spaceships(position, other_spaceships, dims)
  else:
    v_spaceships = np.zeros(dims, dtype=np.float32)
    
  if mode in ('meteoroid', 'all'):
    v_meteoroids = get_velocity_to_avoid_meteoroids(position, meteoroids, dims)
  else:
    v_meteoroids = np.zeros(dims, dtype=np.float32)
    
  v = v_goal + v_planets + v_spaceships + v_meteoroids
  return cap(v, max_speed=max_speed)


def spaceships_update_pose(spaceships, goal_positions, planet_positions, planet_radii, meteoroids, dims, max_speed, dt, mode='all'):
  # MISSING: Need to compute the updated pose of all the spaceships
  # passed through the list spaceships

  # all the passed variables are defined the same was as in the potential fields/ velocity fields
  # dt is the timestep for the update.

  # This is a placeholder for euler simulation
  for i in range(np.shape(spaceships)[0]):
    spaceships[i, :dims] = spaceships[i, :dims] + dt * get_velocity(spaceships[i, :dims], goal_positions[i], planet_positions, planet_radii, np.delete(spaceships, i, 0), meteoroids, dims, max_speed, mode)

  return spaceships


""" Helper functions for entity classes """

def get_obstacle_field_velocity(self, pos, obstacle_pos, obstacle_radius):
    return np.zeros(pos.size) # TODO

def get_goal_field_velocity(self, pos, obstacle_pos, obstacle_radius):
    return np.zeros(pos.size) # TODO

def at_goal(self, pos, goal):
    return False # TODO

def entities_collide(entity1, entity2):
    return (np.hypot(entity1.pos - entity2.pos) <= entity1.radius + entity2.radius)


""" Entity classes """

class Entity:
    def __init__(self, pos, radius):
        # Each entity has a bounding sphere/circle
        self.pos = pos
        self.radius = radius
    def get_obstacle_field_velocity(self, pos):
        raise NotImplementedError("Entity subclass must implement 'update_obstacle_field_velocity'")
    def update(self, dt):
        raise NotImplementedError("Entity subclass must implement 'update'")


# For planets and meteoroids. Meteroids are destructable
class Obstacle(Entity):
    def __init__(self, pos, radius, velocity, destructable=False):
        super().__init__(pos, radius)
        self.velocity = velocity
        self.destructable = destructable
        self.destroy_flag = False

    def get_obstacle_field_velocity(self, pos):
        return get_obstacle_field_velocity(pos, self.pos, self.radius)

    def update(self, dt):
        self.pos += self.velocity * dt

    def in_collision_state(self):
        if (self.destructable):
            self.destoy_flag = True

class Pose:
    def __init__(self, pos, epsilon=1):
        self.pos = pos
        if len(pos.shape) == 2:
            self.A = np.array([
                [1, 0],
                [0, 1/epsilon]
            ])
        else:
            self.A = np.array([
                [1, 0, 0],
                [0, 0, -1/epsilon],
                [0, 1/epsilon, 0]
            ])
        # Use the rotation matrix to encode orientation. Better generalises
        # to 2D and 3D.
        self.R = self.eye(len(pos.shape))

    def move_using_holonomic_point(self, velocity, max_speed, max_angular_speed, dt):
        if len(self.pose.shape) == 2:
            self.R = np.array([
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)]
            ])

        pose_vel = np.matmul(self.A, np.matmul(self.R, velocity))

        linear_vel = pose_vel[0] * self.R[:,0]
        if np.linalg.norm(linear_vel) > max_speed:
            linear_vel *= max_speed/np.linalg.norm(max_speed)
        self.pos += dt * linear_vel

        if len(self.pose.shape) == 2:
            S = np.array([[0, -1], [1, 0]])
            dtheta = pose_vel[1]
            if (dtheta==0): return
        else:
            theta_hat = np.matmul(self.R, np.array([0, pose_vel[1], pose_vel[2]]))
            dtheta = np.linalg.norm(theta_hat)
            if (dtheta==0): return
            theta_hat /= dtheta
            S = np.array([
                [0, -theta_hat[2], theta_hat[1]],
                [theta_hat[2], 0, -theta_hat[0]],
                [-theta_hat[1], theta_hat[0], 0]
            ])
        if abs(dtheta) > max_angular_speed:
            dtheta = max_angular_speed * np.sign(dtheta)
        self.R += S*np.sin(dtheta*dt) + np.matmul(S,S)*(1 - np.cos(dtheta*dt))

    def point_towards(self, pos):
        displacement = pos - self.pos
        if len(self.pos.shape) == 2:
            angle = np.atan2(displacement[1], displacement[0])
            self.R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        else:
            r1 = displacement/np.linalg.norm(displacement)
            r2 = r1
            r2[2] = 0
            if (np.linalg.norm(r2) == 0):
                r2[0] = 1
            else:
                angle = np.arcsin(r1[2]) - np.pi/2
                r2 *= np.cos(angle)/np.linalg.norm(r2)
                r2[2] = np.sin(angle)
            r3 = np.cross(r1, r2)
            self.R[:, 0] = r1
            self.R[:, 1] = r2
            self.R[:, 2] = r3

class Agent(Entity):
    def __init__(self, pos, radius, max_speed, max_angular_speed):
        super().__init__(pos, radius)
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.velocity = np.zeros(pos.shape)
        self.goal = None
        self.pose = Pose(self.pos)

    def get_obstacle_field_velocity(self, pos):
        return get_obstacle_field_velocity(pos, self.pos, self.radius)

    def update(self, dt):
        self.pose.move_using_holonomic_point(self.velocity, self.max_speed, self.max_angular_speed, dt)
        self.pos = self.pose.pos
        if at_goal(self.pos, self.goal):
            self.goal is None

    def set_goal(self, goal):
        self.goal = goal

    def set_velocity(self, obstacle_velocity):
        self.velocity = velocity
        if self.goal is not None:
            self.velocity += get_goal_field_velocity(self.pos, self.goal)

    def point_towards(self, point):
        self.pose.point_towards(point)


class World:
    def __init__(self, num_dimensions, width):
        self.size = np.full(num_dimensions, width)
        self.agents = []
        self.obstacles = []

    def update(self, dt):
        for agent in agents:
            velocity = np.zeros(self.size)
            for obstacle in obstacles:
                velocity += get_obstacle_field_velocity(agent.position)
            for other_agent in agents:
                if other_agent != agent:
                    velocity += other_agent.get_velocity(agent.position)
            agent.set_velocity(velocity)
        for agent in agents:
            agent.update(dt)
        for obstacle in obstacles:
            obstacle.update(dt)

        # Now remove destructable obstacles in a collision state
        for i in range(len(obstacles)):
            for j in range(i+1, len(obstacles)):
                if entities_collide(obstacles[i], obstacles[j]):
                    obstacles[i].in_collision_state()
                    obstacles[j].in_collision_state()
        i = 0
        while i != len(obstacles):
            if obstacles[i].collision_flag:
                obstacles.pop(i)
            else:
                i+=1


# Default Arena
DIMENSIONS = 2
MAX_SPEED = .5
SIZE_OF_UNIVERSE = 10.
PLANET_POSITION = np.array([3., 2.], dtype=np.float32)
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
      velocity = get_velocity(np.array([X[i, j], Y[i, j]]), GOAL_POSITION, np.array(PLANET_POSITION).reshape((-1, DIMENSIONS)), np.array(PLANET_RADIUS).reshape((-1, 1)), np.array(STATIONARY_SPACESHIP).reshape((-1, DIMENSIONS*2)), np.array(METEOROID).reshape((-1, DIMENSIONS*2)), DIMENSIONS, MAX_SPEED, mode='all')
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
    v = get_velocity(x, GOAL_POSITION, np.array(PLANET_POSITION).reshape((-1, DIMENSIONS)), np.array(PLANET_RADIUS).reshape((-1, 1)), np.array(STATIONARY_SPACESHIP).reshape((-1, DIMENSIONS*2)), np.array(METEOROID).reshape((-1, DIMENSIONS*2)), DIMENSIONS, MAX_SPEED, mode='all')
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
