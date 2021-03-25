import potential_field
import numpy as np
import matplotlib.pylab as plt
from PIL import Image


# Name of gif file that we save
name = 'Debugging'

# Defining the environment
DIMENSIONS = int(2)
MAX_SPEED = .5
SIZE_OF_UNIVERSE = 10.
PLANET_POSITIONS = [np.array([3., 2., .5][:DIMENSIONS], dtype=np.float32)]
PLANET_RADII = [.3]
METEOROIDS = [np.concatenate((np.array([-.3, 4.], dtype=np.float32),  MAX_SPEED * 5 * potential_field.normalize(np.array([.3, -4.], dtype=np.float32))))]

# Defining all the spaceships that we have performing gradient
# descent of the potentials
SPACESHIPS = [np.concatenate((np.array([-.3, -4.], dtype=np.float32),  potential_field.normalize(np.array([.3, -4.], dtype=np.float32))))]
GOAL_POSITIONS = [np.array([2.5, 2.5], dtype=np.float32)]


# Defining the simulation details
dt = 0.01
dt = 0.1
simTime = 10.

# List of frames that we store to make a gif
frames = []

# Plot a simple trajectory from the start position to save as a gif
# Uses Euler integration for the ships and the meteoroids
METEOROIDS = np.array(METEOROIDS).reshape((-1, DIMENSIONS*2))
SPACESHIPS = np.array(SPACESHIPS)
GOAL_POSITIONS = np.array(GOAL_POSITIONS)

# Storing the positions of the spaceships for plotting a trace
# of their path
trace = True

if DIMENSIONS == 2:
    if trace is True:
        positions_x = [SPACESHIPS[:, 0].item()]
        positions_y = [SPACESHIPS[:, 1].item()]
    for t in np.arange(0., simTime, dt):
        print(t * 100 / simTime, '%')

        # Updating the positions of the spaceships
        SPACESHIPS = potential_field.spaceships_update_pose(SPACESHIPS, GOAL_POSITIONS, PLANET_POSITIONS, PLANET_RADII, METEOROIDS, DIMENSIONS, MAX_SPEED, dt, mode='all')
        if trace is True:
            positions_x.append(SPACESHIPS[:, 0].item())
            positions_y.append(SPACESHIPS[:, 1].item())

        # Updating the positions of the meteoroids
        METEOROIDS[:, :DIMENSIONS] = METEOROIDS[:, :DIMENSIONS] + dt * METEOROIDS[:, DIMENSIONS:]

        fig, ax = plt.subplots()
        # Ploting velocity/potential field for the first spaceship
        X, Y = np.meshgrid(np.linspace(-SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2, 30),
                            np.linspace(-SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2, 30))
        U = np.zeros_like(X)
        V = np.zeros_like(X)
        for i in range(len(X)):
            for j in range(len(X[0])):
                velocity = potential_field.get_velocity(np.array([X[i, j], Y[i, j]]), GOAL_POSITIONS[0, :], PLANET_POSITIONS, PLANET_RADII, SPACESHIPS[1:, :], METEOROIDS, DIMENSIONS, MAX_SPEED, mode='all')
                U[i, j] = velocity[0]
                V[i, j] = velocity[1]
        plt.quiver(X, Y, U, V, units='width')

        # Ploting planets.
        for i in range(len(PLANET_POSITIONS)):
            ax.add_artist(plt.Circle(PLANET_POSITIONS[i], PLANET_RADII[i], color='gray'))
        
        # Plotting the edges of the universe
        plt.plot([-SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2], [-SIZE_OF_UNIVERSE/2, -SIZE_OF_UNIVERSE/2], 'k')
        plt.plot([-SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2], [SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2], 'k')
        plt.plot([-SIZE_OF_UNIVERSE/2, -SIZE_OF_UNIVERSE/2], [-SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2], 'k')
        plt.plot([SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2], [-SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2], 'k')

        # Plotting the spaceships with a trail
        for i in range(np.shape(SPACESHIPS)[0]):
            if i == 0:
                color = 'g'
            else:
                color = 'b'
            plt.plot(SPACESHIPS[i, 0], SPACESHIPS[i, 1], lw=2, c=color, marker = 'o')
            if trace is True:
                plt.plot(positions_x, positions_y, lw=2, c=color)

        # Plotting meteoroids
        for i in range(np.shape(METEOROIDS)[0]):
            plt.plot(METEOROIDS[i, 0], METEOROIDS[i, 1], c='r', lw=2, marker = 'h')

        # Formatting axes and forcing to plot
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-.5 - SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2 + .5])
        plt.ylim([-.5 - SIZE_OF_UNIVERSE/2, SIZE_OF_UNIVERSE/2 + .5])
        fig.canvas.draw()

        # Adding an image of the plot to make into a gif
        frames.append(Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb()))

        # Closing the figure to free up memory
        plt.close()


# Saving the gif
frames[0].save("{}.gif".format(name), save_all=True, append_images=frames[1:], duration=dt*1000, loop=0)