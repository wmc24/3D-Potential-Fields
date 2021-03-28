import matplotlib.pylab as plt
import numpy as np
import pygame as pg

from camera import Camera2D, Camera3D

class Window:
    def __init__(self, title, width, height, world):
        pg.init()
        pg.display.set_caption(title)
        self.surface = pg.display.set_mode((width, height))
        self.width = width
        self.height = height
        self.world = world

        if self.world.N == 2:
            self.camera = Camera2D(np.array([width/2, height/2]))
            self.camera.scale = 2
        else:
            sef.camera = Camera3D(np.array([width/2, height/2]))

        self.paused = False
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

    def set_clicked_goal(self, mouse_pos):
        if self.active_agent is None: return
        pos = self.camera.untransform_position(np.array(mouse_pos))
        resource_goals = self.world.resource_goals[self.active_agent.resource]
        min_dist = None
        for i, goal in enumerate(resource_goals):
            dist = np.linalg.norm(pos - goal.pos)
            if dist < 100 and (min_dist is None or dist < min_dist):
                min_dist = dist
                self.active_agent.goal = goal
                self.active_agent.last_goal_i = i

    def update(self, dt):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return True
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return True
                elif event.key == pg.K_p:
                    self.paused = not self.paused
                else:
                    self.camera.set_key(event.key, 1)
            if event.type == pg.KEYUP:
                self.camera.set_key(event.key, 0)
            if event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.set_active_agent(event.pos)
                elif event.button == 3:
                    self.set_clicked_goal(event.pos)

        if not self.paused:
            self.camera.update(dt)
            self.world.update(dt)

        return False

    def draw(self):
        self.surface.fill("#000000", pg.Rect(0, 0, self.width, self.height))
        if self.paused:
            self.draw_vfield()
        self.draw_entities()
        pg.display.update()

    def draw_vfield(self):
        speed = 100
        if self.active_agent is not None:
            speed = self.active_agent.max_speed

        if self.paused:
            N = 40
            X, Y = np.meshgrid(np.linspace(0, self.width, N), np.linspace(0, self.height, N))
            U = np.zeros_like(X)
            V = np.zeros_like(Y)
            for i, j in np.ndindex((N, N)):
                screen_pos = np.array([X[i, j], Y[i, j]])
                pos = self.camera.untransform_position(screen_pos)
                velocity = self.world.get_velocity_field(pos, speed, self.active_agent)
                U[i, j] = velocity[0]
                V[i, j] = velocity[1]

            mean_speed = np.sqrt(np.mean(U**2 + V**2))

            for i, j in np.ndindex((N, N)):
                screen_pos = np.array([X[i, j], Y[i, j]])
                velocity = np.array([U[i, j], V[i, j]])
                line_speed = np.linalg.norm(velocity)
                strength = 0.5 * line_speed / mean_speed
                if strength > 1:
                    strength = 1

                line = self.camera.transform_direction(velocity)
                line /= np.linalg.norm(line)
                line_perp = np.array([-line[1], line[0]])

                color = (255*strength, 255*strength, 255*strength)

                lw = 1.0 # line width
                ll = 20.0 # line length
                lhw = 6.0 # Line head width
                lhl = 6.0 # line head length
                points = [
                    screen_pos + line_perp*lw/2,
                    screen_pos + line_perp*lw/2 + line*(ll-lhl),
                    screen_pos + line_perp*lhw/2 + line*(ll-lhl),
                    screen_pos + line*ll,
                    screen_pos - line_perp*lhw/2 + line*(ll-lhl),
                    screen_pos - line_perp*lw/2 + line*(ll-lhl),
                    screen_pos - line_perp*lw/2,
                ]
                pg.draw.polygon(self.surface, color, points)

    def draw_entities(self):
        for resource, goals in self.world.resource_goals.items():
            for i, goal in enumerate(goals):
                pos = self.camera.transform_position(goal.pos)
                direction = self.camera.transform_direction(goal.direction)
                color = self.world.resource_colors[resource]
                radius = self.camera.transform_size(goal.close_dist/2)
                dir_length = self.camera.transform_size(50)
                pg.draw.circle(self.surface, color, pos, radius)
                pg.draw.line(self.surface, color,
                    pos, pos+direction*dir_length, 2)

        for obstacle in self.world.obstacles:
            pos = self.camera.transform_position(obstacle.pos)
            radius = self.camera.transform_size(obstacle.radius)
            pg.draw.circle(self.surface, "#999999", pos, radius)

        for agent in self.world.agents:
            pos = self.camera.transform_position(agent.pos)
            direction = self.camera.transform_direction(agent.pose.get_direction())
            velocity_line = 0.1 * direction * np.linalg.norm(agent.velocity)

            radius = self.camera.transform_size(agent.radius)
            color = self.world.resource_colors[agent.resource]

            pg.draw.circle(self.surface, color, pos, radius)
            pg.draw.line(self.surface, color, pos, pos + velocity_line)

            if agent == self.active_agent:
                outer_radius = self.camera.transform_size(agent.radius+10)
                pg.draw.circle(self.surface, self.world.resource_colors[agent.resource], pos, outer_radius, 2)

            if agent.goal is not None:
                goal_pos = self.camera.transform_position(agent.goal.pos)
                goal_direction = self.camera.transform_direction(agent.goal.direction)
                close_radius = self.camera.transform_size(goal.close_dist)
                pg.draw.circle(self.surface, color, goal_pos, close_radius, 2)

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
            color = self.world.resource_colors[agent.resource]
            ax.add_artist(plt.Circle(agent.pos, agent.radius, color=color))
            log_poses = agent.get_log_poses()
            plt.plot(log_poses[0,:], log_poses[1,:], color=color)
            if agent.goal is not None:
                # Only plot active goals
                plt.plot(agent.goal.pos[0], agent.goal.pos[1], color=color, marker="x")

        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([corner1[0], corner2[0]])
        plt.ylim([corner2[1], corner1[1]])
        plt.show()

