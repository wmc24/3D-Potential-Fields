import numpy as np
import matplotlib.pylab as plt
import pygame as pg

from .geometry import Pose2D, Pose3D, Goal


def entities_collide(entity1, entity2):
    return (np.linalg.norm(entity1.pos - entity2.pos) <= entity1.radius + entity2.radius)

def get_random_meteoroid_trajectory(size, padding=0):
    N = len(size)
    direction = np.zeros(N)
    side = np.random.randint(N)
    sign = -1 if np.random.randint(2)==0 else 1
    direction[side] = sign
    pos = 0.5*direction*size[side]
    for i in range(N):
        if i==side: continue
        pos[i] += (-0.2+0.4*np.random.random())*size[i] + sign*padding
    direction = -direction
    for i in range(N):
        if i==side: continue
        angle = -0.2+0.4*np.random.random() # About -30 -> 30 degrees
        direction[i] = np.sin(angle)
    return pos, direction

def obstacle_outside_bounds(pos, vel, size, padding=0):
    for i in range(len(size)):
        if pos[i] < -size[i]/2-padding and vel[i] < 0: return True
        if pos[i] > size[i]/2+padding and vel[i] > 0: return True
    return False

class Entity(object):
    def __init__(self, pos, radius):
        # Each entity has a bounding sphere/circle
        self.pos = pos
        self.radius = radius
    def update(self, dt):
        raise NotImplementedError("Entity subclass must implement 'update'")


# For planets and meteoroids. Meteroids are destructable
class Obstacle(Entity):
    def __init__(self, pos, radius, velocity=None, destructable=False):
        super().__init__(pos, radius)
        if velocity is None:
            velocity = np.zeros_like(pos)
        self.velocity = velocity
        self.destructable = destructable
        self.remove_flag = False

    def update(self, dt):
        self.pos += self.velocity * dt

    def set_remove_flag(self):
        if (self.destructable):
            self.remove_flag = True

class Agent(Entity):
    def __init__(self, pos, radius, max_speed, max_angular_speed, resource, log_size=0):
        super().__init__(pos, radius)
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.velocity = np.zeros(pos.shape)
        self.goal = None
        self.last_goal_i = -1
        if self.pos.size == 2:
            self.pose = Pose2D(self.pos)
        else:
            self.pose = Pose3D(self.pos)
        self.resource = resource

        self.log_i = 0
        self.log_full = False
        self.log_timer = 0
        if self.pos.size==2:
            self.log_poses = np.zeros((3, log_size))
        else:
            self.log_poses = np.zeros((6, log_size))

    def update(self, dt):
        perturbed_vel = self.velocity + np.random.normal(0, 0.05)
        self.pose.move_using_holonomic_point(perturbed_vel, self.max_speed, self.max_angular_speed, dt)
        self.pos = self.pose.pos
        if self.goal is not None and self.goal.reached(self.pose):
            self.goal = None
            # Also clear logged poses
            self.log_i = 0
            self.log_full = False
            self.log_timer = 0
            self.lot_poses = np.zeros_like(self.log_poses)

        self.log_timer += dt
        if self.log_timer >= 0.1:
            self.log_pose()
            self.log_timer = 0

    def point_towards(self, point):
        self.pose.point_towards(point)

    def log_pose(self):
        if self.log_poses.shape[1] == 0: return
        vec = self.pose.get_vector()
        self.log_poses[0,self.log_i] = vec[0]
        self.log_poses[1,self.log_i] = vec[1]
        self.log_poses[2,self.log_i] = vec[2]
        self.log_i = (self.log_i+1)%self.log_poses.shape[1]
        if not self.log_full and self.log_i == 0:
            self.log_full = True

    def get_log_poses(self):
        if self.log_full:
            return np.concatenate([self.log_poses[:, self.log_i:], self.log_poses[:, :self.log_i]], axis=1)
        else:
            return self.log_poses[:, :self.log_i]

class World:
    def __init__(self, N, width, vfields, log_size=100):
        self.N = N
        self.size = np.full(N, width)
        self.vfields = vfields
        self.agents = []
        self.obstacles = []
        self.meteoroid_spawn_rate = 1
        self.meteoroid_spawn_timer = 0
        self.meteoroid_enable = False
        self.set_meteoroid_spawn_time()
        self.log_size = log_size
        self.resource_colors = {}
        self.resource_goals = {}

    def valid_position(self, pos, radius):
        for obstacle in self.obstacles:
            if np.linalg.norm(obstacle.pos - pos) < obstacle.radius + radius:
                return False
        return True

    def add_planet(self, pos, radius):
        self.obstacles.append(Obstacle(pos,radius))

    def add_agent(self, pos, radius, max_speed, max_angular_speed, resource, color):
        self.agents.append(Agent(pos, radius, max_speed, max_angular_speed, resource, self.log_size))
        self.resource_colors[resource] = color

    def add_goal(self, planet_i, resource):
        if not resource in self.resource_goals:
            self.resource_goals[resource] = []

        angle = np.random.random()*2*np.pi
        direction = np.array([np.cos(angle), np.sin(angle)])
        planet = self.obstacles[planet_i]
        if self.N==3:
            vert_angle = (-0.5+np.random.random())*np.pi
            direction *= np.cos(vert_angle)
            direction = np.array([direction[0], direction[1], np.sin(vert_angle)])

        pos = planet.pos + (planet.radius+40)*direction
        self.resource_goals[resource].append(Goal(pos, -direction))

    def get_velocity_field(self, pos, speed, agent=None):
        velocity = np.zeros(len(self.size))
        for obstacle in self.obstacles:
            if np.linalg.norm(obstacle.velocity) == 0:
                velocity += self.vfields.obstacle(pos, obstacle.pos, obstacle.radius)
            else:
                velocity += self.vfields.moving_obstacle(pos, obstacle.pos, obstacle.radius, obstacle.velocity)
        for other_agent in self.agents:
            if agent is None or agent != other_agent:
                velocity += self.vfields.obstacle(pos, other_agent.pos, other_agent.radius)
        if agent is not None and agent.goal is not None:
            velocity += self.vfields.goal(pos, agent.goal)

        velocity *= speed
        return velocity

    def set_new_goal(self, agent):
        if not agent.resource in self.resource_goals: return
        resources = self.resource_goals[agent.resource]
        if (len(resources) < 2): return

        indexes = [i for i in range(len(resources)) if i != agent.last_goal_i]
        i = np.random.randint(len(indexes))
        agent.goal = resources[indexes[i]]
        agent.last_goal_i = indexes[i]

    def set_meteoroid_spawn_time(self):
        self.meteoroid_spawn_time = -np.log(np.random.random())*self.meteoroid_spawn_rate

    def spawn_meteoroid(self):
        radius = 20 + np.random.random()*20
        pos, direction = get_random_meteoroid_trajectory(self.size, radius)
        speed = 600
        self.obstacles.append(Obstacle(pos, radius, speed*direction, True))

    def update(self, dt):
        # Randomly spawn meteoroids
        if self.meteoroid_enable:
            self.meteoroid_spawn_timer += dt
            if self.meteoroid_spawn_timer > self.meteoroid_spawn_time:
                self.spawn_meteoroid()
                self.meteoroid_spawn_timer = 0
                self.set_meteoroid_spawn_time()

        for agent in self.agents:
            if agent.goal is None:
                self.set_new_goal(agent)
            agent.velocity = self.get_velocity_field(agent.pos, agent.max_speed, agent)

        for agent in self.agents:
            agent.update(dt)
        for obstacle in self.obstacles:
            obstacle.update(dt)

        # Now remove destructable obstacles in a collision state or
        # outside the arena (and moving out the arena)
        for i in range(len(self.obstacles)):
            if obstacle_outside_bounds(
                    self.obstacles[i].pos,
                    self.obstacles[i].velocity,
                    self.size,
                    self.obstacles[i].radius):
                self.obstacles[i].set_remove_flag()
            for j in range(i+1, len(self.obstacles)):
                if entities_collide(self.obstacles[i], self.obstacles[j]):
                    self.obstacles[i].set_remove_flag()
                    self.obstacles[j].set_remove_flag()
                    continue
        i = 0
        while i != len(self.obstacles):
            if self.obstacles[i].remove_flag:
                self.obstacles.pop(i)
            else:
                i+=1

    def log_agent_poses(self, i):
        for agent in agents:
            agent.log_pose(i)
