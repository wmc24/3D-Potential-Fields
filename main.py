import numpy as np
import pygame as pg

from vfields import AnalyticalVFields
from world import World
from window import Window

def main():
    vfields = AnalyticalVFields(
        decay_radius=20,
        convergence_radius=10,
        obstacle_scale=5,
        alpha=10
    )

    world = World(2, 1000, vfields, 100)

    world.add_planet(np.array([-300, -300], dtype=np.float32), 100)
    world.add_planet(np.array([0, 200], dtype=np.float32), 80)
    world.add_planet(np.array([400, -100], dtype=np.float32), 80)

    world.add_agent(
        pos = np.array([100, 0], dtype=np.float32),
        radius = 20,
        max_speed = 3000,
        max_angular_speed = 2,
        resource = "A",
        color = "#ff0000")
    world.add_agent(
        pos = np.array([0, 0], dtype=np.float32),
        radius = 15,
        max_speed = 5000,
        max_angular_speed = 4,
        resource = "B",
        color = "#00ff00")
    world.add_agent(
        pos = np.array([-100, 0], dtype=np.float32),
        radius = 12,
        max_speed = 7000,
        max_angular_speed = 6,
        resource = "C",
        color = "#0000ff")

    world.add_goal(0, "A")
    world.add_goal(0, "B")
    world.add_goal(0, "C")
    world.add_goal(1, "A")
    world.add_goal(1, "B")
    world.add_goal(1, "C")
    world.add_goal(2, "A")
    world.add_goal(2, "B")
    world.add_goal(2, "C")

    window = Window("Spaceships", 1200, 800, world)

    clock = pg.time.Clock()
    while True:
        dt = clock.tick()/1000.0
        if window.update(dt):
            break
        window.draw()
    pg.quit()

    window.plot()

if __name__ == '__main__':
    main()

