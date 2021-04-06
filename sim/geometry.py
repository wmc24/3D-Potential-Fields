import numpy as np

class Pose2D(object):
    def __init__(self, pos, epsilon=10.0):
        assert(pos.size==2)
        self.pos = pos
        self.A = np.array([[1, 0], [0, 1/epsilon]])
        self.angle = 0
        self.R = np.eye(2, dtype=np.float32)

    def get_direction(self):
        return self.R[:,0]

    def move_using_holonomic_point(self, velocity, max_speed, max_angular_speed, dt):
        pose_vel = np.matmul(self.A, np.matmul(self.R.transpose(), velocity))

        linear_vel = pose_vel[0] * self.R[:,0]
        if np.linalg.norm(linear_vel) > max_speed:
            linear_vel *= max_speed/np.linalg.norm(max_speed)
        self.pos += dt * linear_vel

        angular_vel = pose_vel[1]
        if abs(angular_vel) > max_angular_speed:
            angular_vel = np.sign(angular_vel) * max_angular_speed
        self.angle += angular_vel * dt
        while (self.angle < -np.pi): self.angle += 2*np.pi
        while (self.angle > np.pi): self.angle -= 2*np.pi
        self.update_R()

    def update_R(self):
        self.R = np.array([
            [np.cos(self.angle), -np.sin(self.angle)],
            [np.sin(self.angle), np.cos(self.angle)]
        ])

    def get_vector(self):
        return np.array([self.pos[0], self.pos[1], self.angle])


class Pose3D(object):
    def __init__(self, pos, epsilon=10.0):
        assert(pos.size==3)
        self.pos = pos
        self.A = np.array([
            [1, 0, 0],
            [0, 0, -1/epsilon],
            [0, 1/epsilon, 0]
        ])
        self.R = np.eye(3, dtype=np.float32)

    def get_direction(self):
        return self.R[:,0]

    def move_using_holonomic_point(self, velocity, max_speed, max_angular_speed, dt):
        pose_vel = np.matmul(self.A, np.matmul(self.R.transpose(), velocity))

        linear_vel = pose_vel[0] * self.R[:,0]
        if np.linalg.norm(linear_vel) > max_speed:
            linear_vel *= max_speed/np.linalg.norm(max_speed)
        self.pos += dt * linear_vel

        omega = np.matmul(self.R, np.array([0, pose_vel[1], pose_vel[2]]))
        dtheta = np.linalg.norm(omega)
        if dtheta==0: return
        n = omega / dtheta
        S = np.array([
            [0, -n[2], n[1]],
            [n[2], 0, -n[0]],
            [-n[1], n[0], 0]
        ])

        if abs(dtheta) > max_angular_speed:
            dtheta = max_angular_speed * np.sign(dtheta)
        dtheta *= dt
        self.R = np.matmul(np.eye(3) + S*np.sin(dtheta) + np.matmul(S,S)*(1 - np.cos(dtheta)), self.R)

    def get_vector(self):
        return np.concatenate([self.pos, self.R[:,0]])


class Goal:
    def __init__(self, pos, direction, close_dist=20):
        self.pos = pos
        self.direction = direction
        self.close_dist = close_dist
    def reached(self, pose):
        disp = pose.pos - self.pos
        dist = np.linalg.norm(disp)
        return dist < self.close_dist
        # cos_theta = np.dot(pose.get_direction(), self.direction)
        # if dist < self.close_dist:
        #     print(cos_theta)
        # return dist < self.close_dist and cos_theta > 0.87 # < 30 degrees
