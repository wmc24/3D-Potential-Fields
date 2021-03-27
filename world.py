import numpy as np
import matplotlib.pylab as plt

import potential_fields


""" Helper functions for entity classes """

def entities_collide(entity1, entity2):
    return (np.linalg.norm(entity1.pos - entity2.pos) <= entity1.radius + entity2.radius)

def obstacle_outside_bounds(obstacle, size):
    if obstacle.pos[0] < -obstacle.radius and obstacle.velocity[0] < 0: return True
    if obstacle.pos[0] > size[0]+obstacle.radius and obstacle.velocity[0] > 0: return True
    if obstacle.pos[1] < -obstacle.radius and obstacle.velocity[1] < 0: return True
    if obstacle.pos[1] < size[1]+obstacle.radius and obstacle.velocity[1] > 0: return True
    return False


class Entity(object):
    def __init__(self, pos, radius):
        # Each entity has a bounding sphere/circle
        self.pos = pos
        self.radius = radius
    def get_obstacle_velocity_field(self, pos, speed):
        raise NotImplementedError("Entity subclass must implement 'update_obstacle_field_velocity'")
    def update(self, dt):
        raise NotImplementedError("Entity subclass must implement 'update'")


# For planets and meteoroids. Meteroids are destructable
class Obstacle(Entity):
    def __init__(self, pos, radius, velocity=None, destructable=False):
        super(Obstacle,self).__init__(pos, radius)
        if velocity is None:
            velocity = np.zeros_like(pos)
        self.velocity = velocity
        self.destructable = destructable
        self.remove_flag = False

    def get_obstacle_velocity_field(self, pos, speed):
        return potential_fields.obstacle(pos, self.pos, self.radius, speed)

    def update(self, dt):
        self.pos += self.velocity * dt

    def set_remove_flag(self):
        if (self.destructable):
            self.remove_flag = True

class Pose(object):
    def __init__(self, pos, epsilon=1):
        self.pos = pos
        self.N = pos.size
        if self.N:
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
        self.R = np.eye(self.N, dtype=np.float32)

    def move_using_holonomic_point(self, velocity, max_speed, max_angular_speed, dt):
        pose_vel = np.matmul(self.A, np.matmul(self.R.transpose(), velocity))

        linear_vel = pose_vel[0] * self.R[:,0]
        if np.linalg.norm(linear_vel) > max_speed:
            linear_vel *= max_speed/np.linalg.norm(max_speed)
        self.pos += dt * linear_vel

        if self.N == 2:
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
        if self.N == 2:
            angle = np.arctan2(displacement[1], displacement[0])
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

    def get_vector(self):
        if self.N == 2:
            angle = np.arctan2(self.R[0,1], self.R[0,0])
            return np.array([self.pos[0], self.pos[1], angle])
        else:
            return np.concatenate([self.pos, self.R[:,0]])

class Agent(Entity):
    def __init__(self, pos, radius, max_speed, max_angular_speed, resource, log_size=None):
        super(Agent,self).__init__(pos, radius)
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.velocity = np.zeros(pos.shape)
        self.goal = None
        self.pose = Pose(self.pos)
        self.resource = resource
        self.log_poses = None
        if log_size is not None:
            if self.pos.size==2:
                self.log_poses = np.zeros((3, log_size))
            else:
                self.log_poses = np.zeros((6, log_size))

    def get_obstacle_velocity_field(self, pos, speed):
        return potential_fields.obstacle(pos, self.pos, self.radius, speed)

    def update(self, dt):
        self.pose.move_using_holonomic_point(self.velocity, self.max_speed, self.max_angular_speed, dt)
        self.pos = self.pose.pos
        if self.goal is not None and np.linalg.norm(self.pos - self.goal) < 1: # TODO Use variable for this
            self.goal is None

    def set_goal(self, goal):
        self.goal = goal

    def set_velocity(self, obstacle_velocity):
        self.velocity = obstacle_velocity
        if self.goal is not None:
            self.velocity += potential_fields.goal(self.pos, self.goal, self.max_speed)

    def point_towards(self, point):
        self.pose.point_towards(point)

    def log_pose(self, i):
        if self.log_poses is not None and i < self.log_poses.shape[1]:
            vec = self.pose.get_vector()
            self.log_poses[0,i] = vec[0]
            self.log_poses[1,i] = vec[1]
            self.log_poses[2,i] = vec[2]

class World:
    def __init__(self, num_dimensions, width, log_size=None):
        self.size = np.full(num_dimensions, width)
        self.agents = []
        self.obstacles = []
        self.meteoroid_spawn_rate = 5
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

    def add_agent(self, pos, resource, color):
        radius = 10
        max_speed = 100
        max_angular_speed = 3
        self.agents.append(Agent(pos, radius, max_speed, max_angular_speed, resource, self.log_size))
        self.resource_colors[resource] = color

    def add_goal(self, resource, pos):
        if not resource in resource_goals:
            resource_goals[resource] = []
        resource_goals[resource].append(pos)

    def get_agent_velocity_field(self, agent):
        velocity = np.zeros(len(self.size))
        for obstacle in self.obstacles:
            velocity += obstacle.get_obstacle_velocity_field(agent.pos, agent.max_speed)
        for other_agent in self.agents:
            if agent is None or other_agent != agent:
                velocity += other_agent.get_obstacle_velocity_field(agent.pos, agent.max_speed)
        return velocity

    def get_velocity_field(self, pos, speed):
        velocity = np.zeros(len(self.size))
        for obstacle in self.obstacles:
            velocity += obstacle.get_obstacle_velocity_field(pos, speed)
        for agent in self.agents:
            velocity += agent.get_obstacle_velocity_field(pos, speed)
        return velocity

    def update(self, dt, log_i=None):
        for agent in self.agents:
            agent.set_velocity(self.get_agent_velocity_field(agent))
        for agent in self.agents:
            agent.update(dt)
        for obstacle in self.obstacles:
            obstacle.update(dt)

        # Now remove destructable obstacles in a collision state or
        # outside the arena (and moving out the arena)
        for i in range(len(self.obstacles)):
            if obstacle_outside_bounds(self.obstacles[i], self.size):
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

        if log_i is not None and self.log_size is not None:
            for agent in self.agents:
                agent.log_pose(log_i)

    def log_agent_poses(self, i):
        for agent in agents:
            agent.log_pose(i)

    # Plot velocity field and entity positions at the current state, as well
    # the logged positions
    def plot(self):
        if len(self.size) != 2:
            return

        fig, ax = plt.subplots()

        speed = 100

        N = 30
        X, Y = np.meshgrid(np.linspace(0, self.size[0], N), np.linspace(0, self.size[1], N))
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i, j in np.ndindex((N, N)):
            velocity = self.get_velocity_field(np.array([X[i, j], Y[i, j]]), speed)
            U[i, j] = velocity[0]
            V[i, j] = velocity[1]
        plt.quiver(X, Y, U, V, units="width")

        for obstacle in self.obstacles:
            ax.add_artist(plt.Circle(obstacle.pos, obstacle.radius, color="gray"))
        for agent in self.agents:
            ax.add_artist(plt.Circle(agent.pos, agent.radius, color=self.resource_colors[agent.resource]))
            if agent.log_poses is not None:
                plt.plot(agent.log_poses[0,:], agent.log_poses[1,:], color=self.resource_colors[agent.resource])

        for resource, goals in self.resource_goals.items():
            for goal in goals:
                plt.plot(goal[0], goal[1], self.resource_colors[resource], marker="x")

        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([0, self.size[0]])
        plt.ylim([0, self.size[1]])
        plt.show()
