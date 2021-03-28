import numpy as np

# Implement as classes, so you can pass an object to calculate the velocity
# fields. Makes it easier to swap between analytical and neural net.

class VFields(object):
    def obstacle(self, pos, obstacle_pos, obstacle_radius, speed):
        raise NotImplementedError("VFields child class must implement 'obstacle'")
    def goal(self, pos, goal_pos, speed):
        raise NotImplementedError("VFields child class must implement 'obstacle'")

class AnalyticalVFields(VFields):
    def __init__(self, decay_radius, convergence_radius):
        self.decay_radius = decay_radius
        self.convergence_radius = convergence_radius

    def obstacle(self, pos, obstacle_pos, obstacle_radius, speed):
        displacement = pos - obstacle_pos
        distance = np.linalg.norm(displacement)
        if distance > obstacle_radius*0.95:
            direction = displacement / distance
            return 0.2*speed*direction*np.exp(-(distance-obstacle_radius*0.95)/self.decay_radius)
        else:
            return np.zeros_like(pos)

    def goal(self, pos, goal_pos, goal_direction, speed):
        displacement = goal_pos - pos
        distance = np.linalg.norm(displacement)
        velocity =  speed * displacement / distance

class NeuralNetVFields(VFields):
    def __init__(self):
        pass
    def obstacle(self, pos, obstacle_pos, obstacle_radius, speed):
        pass
    def goal(self, pos, goal_pos, speed):
        pass
