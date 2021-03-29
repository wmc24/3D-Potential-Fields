import numpy as np
import pygame as pg

class Camera2D:
    def __init__(self, screen_size):
        self.screen_centre = screen_size/2
        self.screen_size = screen_size
        self.pos = np.zeros(2, dtype=np.float32)
        self.scale = 1
        self.R = np.eye(2)

        self.keys = {}
        self.keys[pg.K_a] = 0
        self.keys[pg.K_d] = 0
        self.keys[pg.K_s] = 0
        self.keys[pg.K_w] = 0
        self.keys[pg.K_EQUALS] = 0
        self.keys[pg.K_MINUS] = 0

        self.speed = 1000
        self.zoom_speed = 2

        self.velocity = np.zeros(2, dtype=np.float32)
        self.zoom_velocity = 0

    def update(self, dt):
        self.velocity[0] = self.scale*self.speed*(self.keys[pg.K_d] - self.keys[pg.K_a])
        self.velocity[1] = self.scale*self.speed*(self.keys[pg.K_w] - self.keys[pg.K_s])
        self.pos += self.velocity*dt

        self.zoom_velocity = -self.zoom_speed*(self.keys[pg.K_EQUALS] - self.keys[pg.K_MINUS])
        self.scale = np.exp(np.log(self.scale) + self.zoom_velocity*dt)

    def transform_position(self, pos):
        screen_pos = np.matmul(self.R.transpose(), pos-self.pos)/self.scale
        screen_pos[1] = -screen_pos[1]
        return self.screen_centre + screen_pos

    def transform_direction(self, direction):
        screen_direction = np.matmul(self.R.transpose(), direction)
        screen_direction[1] = -screen_direction[1]
        return screen_direction

    def transform_size(self, size):
        return size/self.scale

    def untransform_position(self, screen_pos):
        screen_pos = np.array(screen_pos)
        screen_pos -= self.screen_centre
        screen_pos[1] = -screen_pos[1]
        return self.pos + np.matmul(self.R, screen_pos)*self.scale

    def untransform_direction(self, screen_direction):
        screen_direction = np.array(screen_direction)
        screen_direction[1] = -screen_direction[1]
        return np.matmul(self.R, screen_direction)

    def untransform_size(self, size):
        return size*self.scale

    def set_key(self, key, value):
        if key in self.keys:
            self.keys[key] = value


class Camera3D(object):
    def __init__(self, screen_centre, focal_length=10, clipping_dist=10):
        self.screen_centre = screen_centre
        self.focal_length = focal_length
        self.clipping_dist = clipping_dist
        self.pos = np.zeros(3, dtype=np.float32)
        self.scale = 1
        self.angle = 0
        self.update_R()

        self.keys = {}
        self.keys[pg.K_a] = 0
        self.keys[pg.K_d] = 0
        self.keys[pg.K_s] = 0
        self.keys[pg.K_w] = 0
        self.keys[pg.K_q] = 0
        self.keys[pg.K_e] = 0
        self.keys[pg.K_SPACE] = 0
        self.keys[pg.K_LSHIFT] = 0
        self.keys[pg.K_EQUALS] = 0
        self.keys[pg.K_MINUS] = 0

        self.speed = 1000
        self.rotate_speed = 3
        self.zoom_speed = 2

        self.velocity = np.zeros(3, dtype=np.float32)
        self.zoom_velocity = 0

    def update_R(self):
        # Z is out of the screen towards you
        # X and Y are right and up on the screen
        self.R = np.array([
            [np.sin(self.angle),0,-np.cos(self.angle)],
            [-np.cos(self.angle),0,-np.sin(self.angle)],
            [0,1,0]
        ])
        # Only control yaw, don't bother with pitch

    def update(self, dt):
        self.velocity[0] = self.scale*self.speed*(self.keys[pg.K_w] - self.keys[pg.K_s])
        self.velocity[1] = self.scale*self.speed*(self.keys[pg.K_d] - self.keys[pg.K_a])
        self.velocity[2] = self.scale*self.speed*(self.keys[pg.K_SPACE] - self.keys[pg.K_LSHIFT])

        self.pos += np.matmul(self.R, self.velocity)*dt

        self.angle += self.rotate_speed*(self.keys[pg.K_q] - self.keys[pg.K_e])
        self.update_R()

        self.zoom_velocity = -self.zoom_speed*(self.keys[pg.K_EQUALS] - self.keys[pg.K_MINUS])
        self.scale = np.exp(np.log(self.scale) + self.zoom_velocity*dt)

    def transform_position(self, pos):
        pos_3d = np.matmul(self.R, pos-self.pos)/self.scale
        # Only draw for z < -clipping_dist
        if pos_3d[2] > -self.clipping_dist: return None
        pos_3d *= self.focal_length / pos_3d[2]
        return self.screen_centre + pos_3d[0:2]

    def transform_direction(self, direction):
        screen_direction = np.matmul(self.R, direction)
        return screen_direction[0:2]

    def transform_size(self, size):
        return size/self.scale

    def project_position(self, pos):
        pos_3d = np.array([pos[0], pos[1], -self.focal_length])
        direction = pos_3d / np.linalg.norm(pos_3d)
        pos_3d = self.pos + np.matmul(self.R, pos_3d)*self.scale
        direction = np.matmul(self.R, direction)
        return pos_3d, direction

    def set_key(self, key, value):
        if key in self.keys:
            self.keys[key] = value
