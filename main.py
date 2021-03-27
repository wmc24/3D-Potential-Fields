from world import World
from camera import Camera
import numpy as np
import pygame as pg

def main():
    pg.init()
    pg.display.set_caption("Velocity fields")
    WIDTH = 1200
    HEIGHT = 800
    surface = pg.display.set_mode((WIDTH, HEIGHT))

    world = World(2, 1000)
    world.add_planet(np.array([0, 0], dtype=np.float32), 100)
    world.add_planet(np.array([0, 200], dtype=np.float32), 80)
    world.add_agent(np.array([100, 0], dtype=np.float32), "A", "#ff0000")
    world.add_agent(np.array([-100, 0], dtype=np.float32), "B", "#00ff00")

    camera = Camera(np.array([WIDTH/2, HEIGHT/2], dtype=np.float32))
    camera.scale = 2
    camera_speed = 5

    clock = pg.time.Clock()
    while True:
        dt = clock.tick()/1000.0
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return
            if event.type == pg.KEYDOWN:
                camera.set_key(event.key, 1)
            if event.type == pg.KEYUP:
                camera.set_key(event.key, 0)

        camera.update(dt)
        world.update(dt)

        surface.fill("#000000", pg.Rect(0, 0, WIDTH, HEIGHT))
        world.draw(surface, camera)
        pg.display.update()

    # world.plot()

if __name__ == '__main__':
    main()

