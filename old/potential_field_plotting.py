import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


grid_spacing = 50
colourmap = 'jet'
edgecol = 'k'
edgew = 0.5

# Default Arena
DIMENSIONS = 2
MAX_SPEED = .5
SIZE_OF_UNIVERSE = 10.
PLANET_POSITION = np.array([3., 2., .5][:DIMENSIONS], dtype=np.float32)
PLANET_RADIUS = .3
STATIONARY_SPACESHIP = np.concatenate((np.array([-.3, 4.], dtype=np.float32),  normalize(np.array([.3, -4.], dtype=np.float32))))
METEOROID = np.concatenate((np.array([-.3, 4.], dtype=np.float32),  MAX_SPEED * 2 * normalize(np.array([.3, -4.], dtype=np.float32))))
GOAL_POSITION = np.array([5., 5.], dtype=np.float32)
START_POSITION = np.array([-5., -5.], dtype=np.float32)


""" Plotting the Goal Potential """


def goal_potential(x, y):
    return 0.5 * (x - GOAL_POSITION[0])**2 + 0.5 * (y - GOAL_POSITION[1])**2

x = np.linspace(-SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2, grid_spacing)
y = np.linspace(-SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2, grid_spacing)

X, Y = np.meshgrid(x, y)
Z = goal_potential(X, Y)

fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=colourmap, edgecolor=edgecol, linewidth=edgew)
ax.set_zlabel('Potential')
ax.set_xlabel('x')
ax.set_ylabel('y')


