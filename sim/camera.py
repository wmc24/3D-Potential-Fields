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
        screen_pos = np.matmul(self.R, pos-self.pos)/self.scale
        screen_pos[1] = -screen_pos[1]
        return self.screen_centre + screen_pos

    def transform_direction(self, direction):
        screen_direction = np.matmul(self.R, direction)
        screen_direction[1] = -screen_direction[1]
        return screen_direction

    def transform_size(self, size):
        return size/self.scale

    def untransform_position(self, screen_pos):
        screen_pos = np.array(screen_pos)
        screen_pos -= self.screen_centre
        screen_pos[1] = -screen_pos[1]
        return self.pos + np.matmul(self.R.transpose(), screen_pos)*self.scale

    def untransform_direction(self, screen_direction):
        screen_direction = np.array(screen_direction)
        screen_direction[1] = -screen_direction[1]
        return np.matmul(self.R.transpose(), screen_direction)

    def untransform_size(self, size):
        return size*self.scale

    def set_key(self, key, value):
        if key in self.keys:
            self.keys[key] = value


class Camera3D(object):
    def __init__(self, screen_centre):
        self.screen_centre = screen_centre
        self.pos = np.zeros(3, dtype=np.float32)
        self.R = np.eye(3)
        # TODO
