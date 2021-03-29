import numpy as np
import pygame as pg

import argparse
from vfields import AnalyticalVFields, NeuralNetVFields
from sim import World
from sim import Window

def create_2d_world(vfields):
    world = World(2, 1000, vfields, 100)

    world.add_planet(np.array([-300, -300], dtype=np.float32), 100)
    world.add_planet(np.array([0, 200], dtype=np.float32), 80)
    world.add_planet(np.array([400, -100], dtype=np.float32), 80)

    world.add_agent(
        pos = np.array([100, 0], dtype=np.float32),
        radius = 20,
        max_speed = 2000,
        max_angular_speed = 2,
        resource = "FUEL",
        color = "#ff0000")
    world.add_agent(
        pos = np.array([0, 0], dtype=np.float32),
        radius = 15,
        max_speed = 4000,
        max_angular_speed = 4,
        resource = "METALS",
        color = "#00ff00")
    world.add_agent(
        pos = np.array([-100, 0], dtype=np.float32),
        radius = 12,
        max_speed = 5000,
        max_angular_speed = 6,
        resource = "WATER",
        color = "#0000ff")

    world.add_goal(0, "FUEL")
    world.add_goal(0, "METALS")
    world.add_goal(0, "WATER")
    world.add_goal(1, "FUEL")
    world.add_goal(1, "METALS")
    world.add_goal(1, "WATER")
    world.add_goal(2, "FUEL")
    world.add_goal(2, "METALS")
    world.add_goal(2, "WATER")

    return world

def create_3d_world(vfields):
    pass # TODO

def main():
    parser = argparse.ArgumentParser(description="Runs spaceships simulation")
    parser.add_argument("--N", action="store", type=int, default=2, help="Number of dimensions of world", choices=[2, 3])
    parser.add_argument("--vfields", action="store", default="analytical", help="Method of computing velocity fields", choices=["analytical", "nnet"])
    parser.add_argument("--T", action="store", type=float, default=0, help="Simulation time. T=0 for interactive.")

    args, unknown = parser.parse_known_args()

    if args.vfields == "analytical":
        vfields = AnalyticalVFields(
            decay_radius=20,
            convergence_radius=10,
            obstacle_scale=5,
            alpha=10
        )
    else:
        vfields = NeuralNetVFields() # TODO

    if args.N == 2:
        world = create_2d_world(vfields)
    else:
        world = create_3d_world(vfields)

    if args.T == 0:
        window = Window("Spaceships", 1200, 800, world)
        clock = pg.time.Clock()
        while True:
            dt = clock.tick()/1000.0
            if window.update(dt):
                break
            window.draw()
        pg.quit()

        window.plot()
    else:
        dt = 1e-2
        t = 0
        while t < args.T:
            world.update(dt)
            t += dt

    # TODO: Make the whole world be plotted, not just
    # current view in window. Then make plotting independent
    # of window.

if __name__ == '__main__':
    main()

