import numpy as np
import pygame as pg

import argparse
from vfields import AnalyticalVFields, NeuralNetVFields
from sim import World
from sim import Window
from plotting import plot_world


class WorldConfig:
    def __init__(self, N=2):
        self.N = N
        self.width = 2000
        self.max_speed = 500
        self.max_angular_speed = np.pi*2
        self.agent_radius = 20
        self.planet_radii_range = (60, 200) # Not enforced at the moment
        self.meteoroid_radii_range = (30, 80) # Not enforced ..
        self.meteoroid_speed_range = (3000, 5000) # Not enforced ..
        self.resources = ["Oil", "Iron", "Water"]#, "Copper", "Hydrogen", "Silicon"]
        # pygame can use "#rgb" or (r,g,b)
        # Plotting needs "#rgb", only done for 2D
        # 3D rendering needs (r,g,b)
        # Therefore, use "#rgb" for 2D, (r,g,b) for 3D, as a quick fix
        self.resource_colors_2d = ["#ff0000", "#00ff00", "#0000ff"]#, "#ffff00", "#00ffff", "#ff00ff"]
        self.resource_colors_3d = [(255,0,0), (0,255,0), (0,0,255)]#, (255,255,0), (0,255,255), (255,0,255)]

    def resource_color(self, i):
        if self.N==2:
            return self.resource_colors_2d[i]
        else:
            return self.resource_colors_3d[i]


def create_agents(world, config):
    # Simply spawn in a line at the origin
    x = 0
    for i in range(len(config.resources)):
        pos = np.zeros(world.N)
        pos[0] = x
        x += config.agent_radius*4
        world.add_agent(
            pos=pos,
            radius=config.agent_radius,
            max_speed=config.max_speed,
            max_angular_speed=config.max_angular_speed,
            resource=config.resources[i],
            color=config.resource_color(i))

def create_goals(world, config, num_planets):
    for i in range(num_planets):
        for resource in config.resources:
            world.add_goal(i, resource)

def create_world(vfields, config):
    world = World(N=config.N, width=config.width, vfields=vfields)

    if config.N==2:
        world.add_planet(np.array([-300, -300], dtype=np.float32), 100)
        world.add_planet(np.array([0, 200], dtype=np.float32), 80)
        world.add_planet(np.array([400, -100], dtype=np.float32), 80)
        num_planets = 3
    else:
        world.add_planet(np.array([-300, -300, 0], dtype=np.float32), 100)
        world.add_planet(np.array([-120, 500, 0], dtype=np.float32), 80)
        world.add_planet(np.array([600, -100, 0], dtype=np.float32), 80)
        world.add_planet(np.array([50, 10, -300], dtype=np.float32), 50)
        world.add_planet(np.array([50, -120, 250], dtype=np.float32), 100)
        world.add_planet(np.array([-600, 200, 100], dtype=np.float32), 180)
        num_planets = 6

    create_agents(world, config)
    create_goals(world, config, num_planets)

    return world


def main():
    parser = argparse.ArgumentParser(description="Runs spaceships simulation")
    parser.add_argument("--N", action="store", type=int, default=2, help="Number of dimensions of world", choices=[2, 3])
    parser.add_argument("--vfields", action="store", default="analytical", help="Method of computing velocity fields", choices=["analytical", "nnet"])
    parser.add_argument("--T", action="store", type=float, default=0, help="Simulation time. T=0 for interactive.")

    args, unknown = parser.parse_known_args()

    config = WorldConfig(args.N)

    weights = [0.4, 0.8, 1.2]
    if args.vfields == "analytical":
        vfields = AnalyticalVFields(
            weights=weights,
            decay_radius=30,
            convergence_radius=20,
            alpha=10
        )
    else:
        vfields = NeuralNetVFields(weights)

    world = create_world(vfields, config)

    if args.T == 0:
        window = Window("Spaceships", 1200, 800, world)
        clock = pg.time.Clock()
        while True:
            dt = clock.tick()/1000.0
            if (dt > 0.1): dt = 0 # For when unpausing
            if window.update(dt):
                break
            window.draw()
        pg.quit()
        plot_world(world, window.active_agent)
    else:
        dt = 1e-2
        t = 0
        while t < args.T:
            world.update(dt)
            t += dt
        # Pick an arbitrary active agent.
        plot_world(world, world.agents[0])


if __name__ == '__main__':
    main()

