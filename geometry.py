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

    def point_towards(self, pos):
        displacement = pos - self.pos
        angle = np.arctan2(displacement[1], displacement[0])
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
        return np.concatenate([self.pos, self.R[:,0]])

