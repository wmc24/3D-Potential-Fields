import os
import torch
import myModel
import numpy as np

import matplotlib.pyplot as plt

class VFields(object):
    def __init__(self, weights = [0.3, 0.5, 1]):
        self.weights = weights # List [goal, obstacle, moving_obstacle]

    def obstacle(self, pos, obstacle_pos, obstacle_radius):
        disp = pos - obstacle_pos
        dist = np.linalg.norm(disp)
        return self.weights[1] * (disp/dist) * self._obstacle(dist, obstacle_radius)

    def _obstacle(self, dist, radius):
        raise NotImplementedError("VFields child class must implement '_obstacle'")

    def goal(self, pos, goal):
        disp = pos - goal.pos
        u1 = -goal.direction
        disp1_comp = np.dot(disp, u1)
        disp1 = -disp1_comp * u1
        disp2 = disp - disp1
        disp2_comp = np.linalg.norm(disp2)
        u2 = disp2 / disp2_comp
        relative_disp = np.array([disp1_comp, disp2_comp])
        relative_vel = self._goal(relative_disp)
        return self.weights[0] * (u1*relative_vel[0] + u2*relative_vel[1])

    def _goal(self, disp):
        raise NotImplementedError("VFields child class must implement '_goal'")

    def moving_obstacle(self, pos, obstacle_pos, obstacle_radius, obstacle_vel):
        u1 = obstacle_vel / np.linalg.norm(obstacle_vel)
        disp = pos - obstacle_pos
        disp1_comp = np.dot(u1, disp)
        if disp1_comp > obstacle_radius:
            disp2 = disp - disp1_comp*u1
            disp2_comp = np.linalg.norm(disp2)
            u2 = disp2 / disp2_comp
            relative_disp = np.array([disp1_comp-obstacle_radius, disp2_comp])
            # Use 2*moving_obstacle for a larger margin of error
            side_vel = self._moving_obstacle(relative_disp, 2*obstacle_radius, np.linalg.norm(obstacle_vel))
            return self.weights[2]*u2*side_vel
        return np.zeros_like(pos)

    def _moving_obstacle(self, disp, obstacle_radius, speed):
        raise NotImplementedError("VFields child class must implement '_moving_obstacle'")

class AnalyticalVFields(VFields):
    def __init__(self, weights, decay_radius, convergence_radius, alpha):
        super().__init__(weights)
        self.decay_radius = decay_radius
        self.convergence_radius = convergence_radius
        self.alpha = alpha

    def _obstacle(self, dist, radius):
        if dist >= radius:
            return np.exp(-(dist-radius)/self.decay_radius)
        else:
            return 0

    def _obstacle_list(self, dist, radius):
        # Version that deals with a list/array of displacement values passed
        velocity = np.zeros(len(dist))
        for i in range(len(dist)):
            if dist[i] >= radius:
                velocity[i] = np.exp(-(dist[i]-radius)/self.decay_radius)
            else:
                velocity[i] = 0
        return velocity

    def _goal(self, disp):
        dist = np.linalg.norm(disp)
        if dist < self.convergence_radius:
            velocity = -disp / self.convergence_radius
        else:
            velocity = -disp / dist
        return velocity

    def _goal_list(self, disp):
        # Version that deals with a list/array of displacement values passed
        velocity = np.zeros(np.shape(disp))
        for i in range(len(disp)):
            dist = np.linalg.norm(disp[i, :])
            if dist < self.convergence_radius:
                velocity[i, :] = -disp[i, :] / self.convergence_radius
            else:
                velocity[i, :] = -disp[i, :] / dist
        return velocity

    def _moving_obstacle(self, disp, obstacle_radius, speed):
        if speed==0: return np.array(0)
        urgency = 3 * disp[0] / speed
        if abs(disp[1]) < obstacle_radius:
            avoid_speed = 1
        else:
            avoid_speed = np.exp(-(abs(disp[1])-obstacle_radius)/self.decay_radius)
        side_vel = np.sign(disp[1]) * np.exp(-urgency) * avoid_speed
        return side_vel

    def _moving_obstacle_list(self, disp, obstacle_radius, speed):
        # Version that deals with a list/array of displacement values passed
        velocity = np.zeros(np.shape(disp)[0])
        for i in range(len(disp)):
            if speed!=0:
                disp_i = disp[i, :]
                urgency = 3 * disp_i[0] / speed
                if abs(disp_i[1]) < obstacle_radius:
                    avoid_speed = 1
                else:
                    avoid_speed = np.exp(-(abs(disp_i[1])-obstacle_radius)/self.decay_radius)
                velocity[i] = np.sign(disp_i[1]) * np.exp(-urgency) * avoid_speed
        return velocity



class NeuralNetVFields(VFields):
    def __init__(self, weights, model_name='Goal-one-obstacle-field-for-all'):
        super().__init__(weights)
        #loading the network and its weights
        self.model_name = model_name
        self.model = myModel.Net()
        self.checkpoint = torch.load(os.path.join('models', f'{self.model_name}.tar'), map_location='cpu')
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.whitening = self.checkpoint['whitening']
        self.model.to('cpu')
        self.model.eval()

    def _obstacle(self, dist, radius):
        speed = self.model.forward_obstacle(torch.tensor(dist).float(), torch.tensor(radius).float()).detach().squeeze().numpy()
        return speed

    def _goal(self, disp):
        velocity = self.model.forward_goal(torch.tensor(disp).reshape((1, -1)).float()).detach().squeeze().numpy()
        return velocity

    def _moving_obstacle(self, disp, obstacle_radius, speed):
        velocity = self.model.forward_goal(torch.tensor(disp).reshape((1, -1)).float(), 
                                           torch.tensor(obstacle_radius).reshape((1, -1)).float(), 
                                           torch.tensor(speed).reshape((1, -1)).float()).detach().squeeze().numpy()
        return np.zeros_like(disp)
