import numpy as np
import pygame as pg

class Camera:
    # Assuming 2D for now
    def __init__(self, screen_centre, N=2):
        self.screen_centre = screen_centre
        self.pos = np.zeros(N, dtype=np.float32)
        self.angle = 0
        self.scale = 1

        self.keys = {}
        self.keys[pg.K_a] = 0
        self.keys[pg.K_d] = 0
        self.keys[pg.K_s] = 0
        self.keys[pg.K_w] = 0
        self.keys[pg.K_EQUALS] = 0
        self.keys[pg.K_MINUS] = 0

        self.speed = 1000
        self.zoom_speed = 2

        self.velocity = np.zeros(N, dtype=np.float32)
        self.zoom_velocity = 0

    def update(self, dt):
        self.velocity[0] = -self.speed*(self.keys[pg.K_d] - self.keys[pg.K_a])
        self.velocity[1] = self.speed*(self.keys[pg.K_w] - self.keys[pg.K_s])
        self.pos += self.velocity*dt

        self.zoom_velocity = -self.zoom_speed*(self.keys[pg.K_EQUALS] - self.keys[pg.K_MINUS])
        self.scale = np.exp(np.log(self.scale) + self.zoom_velocity*dt)

    def transform_circle(self, pos, radius):
        T = np.array([
            [np.cos(self.angle), -np.sin(self.angle), self.pos[0]],
            [np.sin(self.angle), np.cos(self.angle), self.pos[1]],
            [0, 0, self.scale]
        ])

        pos_h = np.concatenate([pos, [1]])
        pos_h = np.matmul(T, pos_h)

        new_pos = pos_h[:-1]/pos_h[-1]
        new_radius = radius / self.scale

        return self.screen_centre+new_pos, new_radius

    def set_key(self, key, value):
        if key in self.keys:
            self.keys[key] = value
