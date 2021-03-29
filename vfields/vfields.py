import numpy as np

class VFields(object):
    def obstacle(self, pos, obstacle_pos, obstacle_radius):
        disp = pos - obstacle_pos
        dist = np.linalg.norm(disp)
        return (disp/dist) * self._obstacle(dist, obstacle_radius)

    def _obstacle(self, dist, radius):
        raise NotImplementedError("VFields child class must implement '_obstacle'")

    def goal(self, pos, goal):
        disp = pos - goal.pos
        u1 = -goal.direction
        disp1_comp = np.dot(disp, u1)
        disp1 = -goal.direction * u1
        disp2 = disp - disp1
        disp2_comp = np.linalg.norm(disp2)
        u2 = disp2 / disp2_comp
        relative_disp = np.array([disp1_comp, disp2_comp])
        relative_vel = self._goal(relative_disp)
        return u1*relative_vel[0] + u2*relative_vel[1]

    def _goal(self, disp):
        raise NotImplementedError("VFields child class must implement '_goal'")


class AnalyticalVFields(VFields):
    def __init__(self, decay_radius, convergence_radius, obstacle_scale, alpha):
        self.decay_radius = decay_radius
        self.convergence_radius = convergence_radius
        self.obstacle_scale = obstacle_scale
        self.alpha = alpha

    def _obstacle(self, dist, radius):
        if dist > radius:
            return self.obstacle_scale*np.exp(-(dist-radius)/self.decay_radius)
        else:
            return 0

    def _goal(self, disp):
        dist = np.linalg.norm(disp)
        if dist < self.convergence_radius:
            velocity = -disp / self.convergence_radius
        else:
            velocity = -disp / dist
        # The below penalises tan(theta)**2, to drive to theta=0.
        # velocity += self.alpha * (disp[0]/disp[1]**3)*np.array([disp[1], -disp[0]])
        return velocity


class NeuralNetVFields(VFields):
    def __init__(self):
        pass
    def _obstacle(self, dist, radius):
        return 0 # TODO
    def _goal(self, disp):
        return np.zeros(2) # TODO