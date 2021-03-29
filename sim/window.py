import matplotlib.pylab as plt
import numpy as np
import pygame as pg

from .camera import Camera2D, Camera3D

class DrawCommandLine:
    def __init__(self, end, thickness):
        self.end = end
        self.thickness = thickness

class DrawCommand:
    def __init__(self, pos, radius, color, thickness=0):
        self.pos = pos
        self.radius = radius
        self.color = color
        self.thickness=thickness
        self.lines = []
    def add_line(self, end, thickness):
        self.lines.append(DrawCommandLine(end, thickness))

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
            self.camera = Camera3D(np.array([width/2, height/2]))

        self.paused = False
        self.active_agent = None

        font = pg.font.SysFont(None, 24)
        self.resource_names = []
        for resource, color in self.world.resource_colors.items():
            self.resource_names.append(font.render(resource, True, color))

        self.meteoroid_enable_imgs = []
        self.meteoroid_enable_imgs.append(font.render("Meteoroids: Disabled", True, "#aaaaaa"))
        self.meteoroid_enable_imgs.append(font.render("Meteoroids: Enabled", True, "#aaaaaa"))

    def get_mouse_dist_function(self, mouse_pos):
        if self.world.N == 2:
            pos = self.camera.untransform_position(np.array(mouse_pos))
            return lambda x: np.linalg.norm(pos - x)
        else:
            pos, direction = self.camera.project_position(np.array(mouse_pos))
            return lambda x: np.linalg.norm(
                (pos-x) - direction*np.dot(direction, pos-x))

    def set_active_agent(self, mouse_pos):
        min_dist = None
        self.active_agent = None
        dist_f = self.get_mouse_dist_function(mouse_pos)
        for i, agent in enumerate(self.world.agents):
            dist = dist_f(agent.pos)
            if dist < 100 and (min_dist is None or dist < min_dist):
                min_dist = dist
                self.active_agent = agent

    def set_clicked_goal(self, mouse_pos):
        if self.active_agent is None: return
        resource_goals = self.world.resource_goals[self.active_agent.resource]
        min_dist = None
        dist_f = self.get_mouse_dist_function(mouse_pos)
        for i, goal in enumerate(resource_goals):
            dist = dist_f(goal.pos)
            if dist < 400 and (min_dist is None or dist < min_dist):
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
                elif event.key == pg.K_o:
                    self.world.meteoroid_enable = not self.world.meteoroid_enable
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
        self.surface.fill((0,0,0), pg.Rect(0, 0, self.width, self.height))
        if self.paused and self.world.N==2:
            self.draw_vfield()
        self.draw_world()
        self.draw_hud()
        pg.display.update()

    def draw_vfield(self):
        speed = 100
        if self.active_agent is not None:
            speed = self.active_agent.max_speed

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
        if mean_speed == 0:
            return

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

    def draw_world(self):
        commands = []
        for resource, goals in self.world.resource_goals.items():
            for i, goal in enumerate(goals):
                pos, radius = self.camera.transform_circle(goal.pos, goal.close_dist/2)
                if pos is None: continue
                color = self.world.resource_colors[resource]
                command = DrawCommand(pos, radius, color)

                line_end = self.camera.transform_position(
                    goal.pos+goal.direction*50)
                if line_end is not None:
                    command.add_line(line_end, 2)

                commands.append(command)

        for obstacle in self.world.obstacles:
            pos, radius = self.camera.transform_circle(obstacle.pos, obstacle.radius)
            if pos is None: continue
            commands.append(DrawCommand(pos, radius, (100,100,100)))

        for agent in self.world.agents:
            pos, radius = self.camera.transform_circle(agent.pos, agent.radius)
            if pos is None: continue
            color = self.world.resource_colors[agent.resource]
            command = DrawCommand(pos, radius, color)

            line_end = agent.pos + agent.pose.get_direction() * 0.3 * np.linalg.norm(agent.velocity)
            line_end = self.camera.transform_position(line_end)
            if line_end is not None:
                command.add_line(line_end, 2)

            commands.append(command)

            if agent == self.active_agent:
                pos, outer_radius = self.camera.transform_circle(agent.pos, agent.radius+10)
                if pos is not None:
                    commands.append(DrawCommand(pos, outer_radius, color, 2))

            if agent.goal is not None:
                pos, radius = self.camera.transform_circle(agent.goal.pos, goal.close_dist)
                if pos is None: continue
                commands.append(DrawCommand(pos, radius, color, 2))

        # If N==3, sort commands by z, and darken further colors
        if self.world.N==3:
            commands = sorted(commands, key=lambda x: x.pos[2], reverse=True)
            for command in commands:
                fade = np.exp(-command.pos[2] / 1000)
                command.color = tuple([int(c*fade) for c in command.color])

        for command in commands:
            pg.draw.circle(self.surface, command.color, command.pos[0:2], command.radius, command.thickness)
            for line in command.lines:
                pg.draw.line(self.surface, command.color, command.pos[0:2], line.end[0:2], line.thickness)


    def draw_hud(self):
        x = 20
        for name_img in self.resource_names:
            self.surface.blit(name_img, (x, 20))
            x += 20 + name_img.get_width()
        met_img = self.meteoroid_enable_imgs[self.world.meteoroid_enable]
        self.surface.blit(met_img, (self.width-20-met_img.get_width(), 20))
