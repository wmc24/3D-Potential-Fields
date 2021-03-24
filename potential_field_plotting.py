import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


grid_spacing = 50
colourmap = 'viridis'
colourmap = 'jet'
edgecol = 'k'
edgew = 0.5


""" Plotting the Goal Potential """


def goal_potential(x, y):
    goal = (1.5, 1.5)
    return 0.5 * (x - goal[0])**2 + 0.5 * (y - goal[1])**2

x = np.linspace(-2, 2, grid_spacing)
y = np.linspace(-2, 2, grid_spacing)

X, Y = np.meshgrid(x, y)
Z = goal_potential(X, Y)

fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=colourmap, edgecolor=edgecol, linewidth=edgew)
ax.set_zlabel('Potential')
ax.set_xlabel('x')
ax.set_ylabel('y')


""" Plotting the Obstacle Potential """


def obstacle_potential(x, y, obstacle=(0.3, 0.2), radius=0.3):
    c = 6 * radius**3 / 2
    potential = c / ((x - obstacle[0])** 2 + (y - obstacle[1])** 2)
    # Capping value when within the obstacle just for
    # plotting purposes
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            if (x[i, j] - obstacle[0])** 2 + (y[i, j] - obstacle[1])** 2 < radius**2:
                potential[i, j] = c / radius** 2
                pass
    return potential

x = np.linspace(-2, 2, grid_spacing)
y = np.linspace(-2, 2, grid_spacing)

X, Y = np.meshgrid(x, y)
Z = obstacle_potential(X, Y)

fig = plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=colourmap, edgecolor=edgecol, linewidth=edgew)
ax.set_zlabel('Potential')
ax.set_xlabel('x')
ax.set_ylabel('y')


""" Plotting the Combined Potential """


x = np.linspace(-2, 2, grid_spacing)
y = np.linspace(-2, 2, grid_spacing)

X, Y = np.meshgrid(x, y)
Z = obstacle_potential(X, Y) + goal_potential(X, Y)

fig = plt.figure(3)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=colourmap, edgecolor=edgecol, linewidth=edgew)
ax.set_zlabel('Potential')
ax.set_xlabel('x')
ax.set_ylabel('y')


""" Plotting the Issue in d) """


x = np.linspace(-2, 2, grid_spacing)
y = np.linspace(-2, 2, grid_spacing)

X, Y = np.meshgrid(x, y)
Z = obstacle_potential(X, Y, obstacle=(0, 0)) + goal_potential(X, Y)

fig = plt.figure(4)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=colourmap, edgecolor=edgecol, linewidth=edgew)
ax.set_zlabel('Potential')
ax.set_xlabel('x')
ax.set_ylabel('y')


""" Plotting the Mitigating solution in e) """


x = np.linspace(-2, 2, grid_spacing)
y = np.linspace(-2, 2, grid_spacing)

X, Y = np.meshgrid(x, y)
Z = obstacle_potential(X, Y, obstacle=(0, 0)) + goal_potential(X, Y) + obstacle_potential(X, Y, obstacle=(-1.5, -1.6), radius=0.1)

fig = plt.figure(5)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=colourmap, edgecolor=edgecol, linewidth=edgew)
ax.set_zlabel('Potential')
ax.set_xlabel('x')
ax.set_ylabel('y')


""" Plotting the Issue in f) """


x = np.linspace(-1, 1, grid_spacing)
y = np.linspace(-1, 1, grid_spacing)

X, Y = np.meshgrid(x, y)
Z = obstacle_potential(X, Y, obstacle=(0.5 ,0)) + obstacle_potential(X, Y, obstacle=(0, 0.5)) + goal_potential(X, Y)

fig = plt.figure(6)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=colourmap, edgecolor=edgecol, linewidth=edgew)
ax.set_zlabel('Potential')
ax.set_xlabel('x')
ax.set_ylabel('y')


""" Plotting the Mitigating solution to f) """


x = np.linspace(-2, 2, grid_spacing)
y = np.linspace(-2, 2, grid_spacing)

X, Y = np.meshgrid(x, y)
Z = obstacle_potential(X, Y, obstacle=(0.5 ,0)) + obstacle_potential(X, Y, obstacle=(0, 0.5)) + goal_potential(X, Y) \
    + obstacle_potential(X, Y, obstacle=(-1.5, -1.6), radius=0.1)  + obstacle_potential(X, Y, obstacle=(0, 0), radius=0.5)

fig = plt.figure(7)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=colourmap, edgecolor=edgecol, linewidth=edgew)
ax.set_zlabel('Potential')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()