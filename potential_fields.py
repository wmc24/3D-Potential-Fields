import numpy as np

def obstacle(pos, obstacle_pos, obstacle_radius, speed, decay_radius=100):
    displacement = pos - obstacle_pos
    distance = np.linalg.norm(displacement)
    if distance > obstacle_radius:
        direction = displacement / distance
        return speed*direction*np.exp(-(distance-obstacle_radius)/decay_radius)
    else:
        return np.zeros_like(pos)

def goal(pos, obstacle_pos, obstacle_radius, speed, convergence_radius=10):
    displacement = obstacle_pos - pos
    distance = np.linalg.norm(displacement)
    if distance < convergence_radius:
        return speed * displacement / convergence_radius
    else:
        return speed * displacement / distance
