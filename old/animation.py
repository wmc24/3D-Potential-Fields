import potential_field
import numpy as np
import matplotlib.pylab as plt
from PIL import Image


# Name of gif file that we save in the folder 'gifs'
name = 'Meteoroid-Collision'
# List of frames that we store to make a gif
frames = []

# Defining the environment
DIMENSIONS = int(2)
MAX_SPEED = .5
SIZE_OF_UNIVERSE = 10.
PLANET_POSITIONS = [np.array([3., 2., .5][:DIMENSIONS], dtype=np.float32)]
PLANET_RADII = [.3]
METEOROIDS = [np.concatenate((np.array([-.3, 4.], dtype=np.float32),  MAX_SPEED * 5 * potential_field.normalize(np.array([.3, -4.], dtype=np.float32))))]

# Defining the spaceships
SPACESHIPS = [np.concatenate((np.array([-.3, -4.], dtype=np.float32),  potential_field.normalize(np.array([.3, -4.], dtype=np.float32)))),
              np.concatenate((np.array([-4.5, 0.], dtype=np.float32),  potential_field.normalize(np.array([-.3, -4.], dtype=np.float32)))),
              np.concatenate((np.array([3., -4.], dtype=np.float32),  potential_field.normalize(np.array([-.3, 4.], dtype=np.float32))))]
GOAL_POSITIONS = [np.array([2.5, 2.5], dtype=np.float32),
                  np.array([-1.5, 0.5], dtype=np.float32),
                  np.array([-2.5, 2.5], dtype=np.float32)]

# Defining the simulation details
dt = 0.01
simTime = 50.

# Converting the passed lists into arrays and reshaping into
# the form specified for the potential fields
SPACESHIPS = np.array(SPACESHIPS).reshape((-1, DIMENSIONS*2))
GOAL_POSITIONS = np.array(GOAL_POSITIONS).reshape((-1, DIMENSIONS))
PLANET_POSITIONS = np.array(PLANET_POSITIONS).reshape((-1, DIMENSIONS))
PLANET_RADII = np.array(PLANET_RADII).reshape((-1, 1))
METEOROIDS = np.array(METEOROIDS).reshape((-1, DIMENSIONS*2))

# Storing the positions of the spaceships for plotting a trace
# of their path if true
trace = True

# Adding objects to indicated a collision between a spaceship
# and another object if true. collision_dist is how close objects
# have to get together for a collision to occur
# We also change the icon for a spaceship if it has collided to
# indicate that it is a wreckage
collisions = True
collision_dist = 0.25
wreckage = np.zeros(np.shape(SPACESHIPS)[0])

if DIMENSIONS == 2:
    if trace is True:
        positions = [SPACESHIPS[:, :DIMENSIONS]]
    counter = 0
    for t in np.arange(0., simTime, dt):
        # Printing a counter to show progress as this is slow
        if t * 100 / simTime % 10 == 0:
            print(int(t * 100 / simTime), '%')

        # Updating the positions of the spaceships
        SPACESHIPS = potential_field.spaceships_update_pose(SPACESHIPS, GOAL_POSITIONS, PLANET_POSITIONS, PLANET_RADII, METEOROIDS, DIMENSIONS, MAX_SPEED, dt, mode='all')
        if trace is True:
            positions.append(SPACESHIPS[:, :DIMENSIONS].copy())
            numpy_positions = np.array(positions)

        # Updating the positions of the meteoroids
        METEOROIDS[:, :DIMENSIONS] = METEOROIDS[:, :DIMENSIONS] + dt * METEOROIDS[:, DIMENSIONS:]

        # Saving a frame for the gif every 0.1 s
        counter += 1
        if counter == 0.1 / dt:
            counter = 0
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

            # Plotting the spaceships and their traces
            for i in range(np.shape(SPACESHIPS)[0]):
                # Setting colour for main spaceship
                if i == 0:
                    color = 'g'
                else:
                    color = 'b'
                # Setting markers for wreckages
                if wreckage[i] == 1:
                    mark = '*'
                else:
                    mark = 'o'
                ship, = plt.plot(SPACESHIPS[i, 0], SPACESHIPS[i, 1], lw=2, marker = mark, c=color)

                if trace is True:
                    plt.plot(numpy_positions[:, i, 0], numpy_positions[:, i, 1], lw=2, c=color)

                # Checking and Plotting collisions between spaceships 
                if collisions is True:
                    for j in range(i+1, np.shape(SPACESHIPS)[0]):
                            if np.sqrt(np.sum((SPACESHIPS[i, :DIMENSIONS] - SPACESHIPS[j, :DIMENSIONS])**2)) <= collision_dist:
                                plt.plot(SPACESHIPS[j, 0], SPACESHIPS[j, 1], lw=2, c='orange', markersize=SIZE_OF_UNIVERSE*2, marker = '*')
                                wreckage[j] = 1
                                wreckage[i] = 1

            # Plotting meteoroids
            for i in range(np.shape(METEOROIDS)[0]):
                plt.plot(METEOROIDS[i, 0], METEOROIDS[i, 1], c='r', lw=2, marker = 'p')
                # Checking and Plotting collisions with spaceships
                if collisions is True:
                    for j in range(np.shape(SPACESHIPS)[0]):
                        if np.sqrt(np.sum((METEOROIDS[i, :DIMENSIONS] - SPACESHIPS[j, :DIMENSIONS])**2)) <= collision_dist:
                            plt.plot(SPACESHIPS[j, 0], SPACESHIPS[j, 1], lw=2, c='orange', markersize=SIZE_OF_UNIVERSE*2, marker = '*')
                            wreckage[j] = 1

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

else:
    print("3D animation hasn't been developed yet")


# Saving the gif
frames[0].save("gifs/{}.gif".format(name), save_all=True, append_images=frames[1:])#, duration=.001, loop=0)
