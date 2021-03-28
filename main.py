import numpy as np
import pygame as pg

from vfields import AnalyticalVFields
from world import World
from window import Window

def main():
    vfields = AnalyticalVFields(
        decay_radius=20,
        convergence_radius=100
    )

    world = World(2, 1000, vfields, 100)

    world.add_planet(np.array([-300, -300], dtype=np.float32), 100)
    world.add_planet(np.array([0, 200], dtype=np.float32), 80)

    world.add_agent(np.array([100, 0], dtype=np.float32), "A", "#ff0000")
    world.add_agent(np.array([-100, 0], dtype=np.float32), "B", "#00ff00")

    world.add_goal(np.array([-100, 200], dtype=np.float32), "A")
    world.add_goal(np.array([300, 200], dtype=np.float32), "A")

    world.add_goal(np.array([-300, 200], dtype=np.float32), "B")
    world.add_goal(np.array([-500, 200], dtype=np.float32), "B")

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

