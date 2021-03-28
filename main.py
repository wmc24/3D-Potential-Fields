import numpy as np
import pygame as pg
import matplotlib.pylab as plt

from world import World
from camera import Camera
from vfields import AnalyticalVFields

class Window:
    def __init__(self, title, width, height, world):
        pg.init()
        pg.display.set_caption(title)
        self.surface = pg.display.set_mode((width, height))
        self.width = width
        self.height = height
        self.world = world

        self.camera = Camera(np.array([width/2, height/2]))
        self.camera.scale = 2

        self.plot_field = False
        self.active_agent = None

    def set_active_agent(self, mouse_pos):
        pos = self.camera.untransform_position(np.array(mouse_pos))
        min_dist = None
        self.active_agent = None
        for i, agent in enumerate(self.world.agents):
            dist = np.linalg.norm(pos - agent.pos)
            if dist < 100 and (min_dist is None or dist < min_dist):
                min_dist = dist
                self.active_agent = agent

    def update(self, dt):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return True
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return True
                elif event.key == pg.K_p:
                    self.plot_field = not self.plot_field
                else:
                    self.camera.set_key(event.key, 1)
            if event.type == pg.KEYUP:
                self.camera.set_key(event.key, 0)
            if event.type == pg.MOUSEBUTTONDOWN:
                self.set_active_agent(event.pos)

        if not self.plot_field:
            self.camera.update(dt)
            self.world.update(dt)

        return False

    def draw(self):
        self.surface.fill("#000000", pg.Rect(0, 0, self.width, self.height))

        for obstacle in self.world.obstacles:
            pos = self.camera.transform_position(obstacle.pos)
            radius = self.camera.transform_size(obstacle.radius)
            pg.draw.circle(self.surface, "#999999", pos, radius)


        for agent in self.world.agents:
            pos = self.camera.transform_position(agent.pos)
            radius = self.camera.transform_size(agent.radius)
            pg.draw.circle(self.surface, self.world.resource_colors[agent.resource], pos, radius)
            if agent == self.active_agent:
                pg.draw.circle(self.surface, self.world.resource_colors[agent.resource], pos, radius+5, 2)

        speed = 100
        if self.active_agent is not None:
            speed = self.active_agent.max_speed

        if self.plot_field:
            N = 50
            X, Y = np.meshgrid(np.linspace(0, self.width, N), np.linspace(0, self.height, N))
            for i, j in np.ndindex((N, N)):
                screen_pos = np.array([X[i, j], Y[i, j]])
                pos = self.camera.untransform_position(screen_pos)
                velocity = self.world.get_velocity_field(pos, speed, self.active_agent)
                velocity = self.camera.transform_direction(velocity) * 0.8
                pg.draw.line(self.surface, "#ffffff", screen_pos, screen_pos+velocity, 2)

        pg.display.update()

    def plot(self):
        if len(self.world.size) != 2:
            return

        fig, ax = plt.subplots()

        speed = 100

        N = 50
        corner1 = self.camera.untransform_position(np.array([0, 0]))
        corner2 = self.camera.untransform_position(np.array([self.width, self.height]))

        X, Y = np.meshgrid(np.linspace(corner1[0], corner2[0], N), np.linspace(corner1[1], corner2[1], N))
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i, j in np.ndindex((N, N)):
            pos = np.array([X[i, j], Y[i, j]])
            velocity = 0.1*self.world.get_velocity_field(pos, speed, self.active_agent)
            U[i, j] = velocity[0]
            V[i, j] = velocity[1]
        plt.quiver(X, Y, U, V, units="width")

        for obstacle in self.world.obstacles:
            ax.add_artist(plt.Circle(obstacle.pos, obstacle.radius, color="gray"))
        for agent in self.world.agents:
            ax.add_artist(plt.Circle(agent.pos, agent.radius, color=self.world.resource_colors[agent.resource]))
            if agent.log_poses is not None:
                plt.plot(agent.log_poses[0,:], agent.log_poses[1,:], color=self.world.resource_colors[agent.resource])

        for resource, goals in self.world.resource_goals.items():
            for goal in goals:
                plt.plot(goal[0], goal[1], self.world.resource_colors[resource], marker="x")

        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([corner1[0], corner2[0]])
        plt.ylim([corner1[1], corner2[1]])
        plt.show()


def main():
    vfields = AnalyticalVFields(
        decay_radius=100,
        convergence_radius=100
    )

    world = World(2, 1000, vfields)
    world.add_planet(np.array([-300, -300], dtype=np.float32), 100)
    world.add_planet(np.array([0, 200], dtype=np.float32), 80)
    world.add_agent(np.array([100, 0], dtype=np.float32), "A", "#ff0000")
    world.add_agent(np.array([-100, 0], dtype=np.float32), "B", "#00ff00")

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

