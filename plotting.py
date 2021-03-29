import matplotlib.pylab as plt
import numpy as np

def plot_world(world, active_agent=None):
    if len(world.size) != 2:
        return

    fig, ax = plt.subplots()

    speed = 100 # Arbitrary

    N = 50
    corner1 = np.array(-world.size/2, dtype=np.float32)
    corner2 = np.array(world.size/2, dtype=np.float32)

    X, Y = np.meshgrid(np.linspace(corner1[0], corner2[0], N), np.linspace(corner1[1], corner2[1], N))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i, j in np.ndindex((N, N)):
        pos = np.array([X[i, j], Y[i, j]])
        velocity = world.get_velocity_field(pos, speed, active_agent)
        U[i, j] = velocity[0]
        V[i, j] = velocity[1]
    plt.quiver(X, Y, U, V, units="width")

    for obstacle in world.obstacles:
        ax.add_artist(plt.Circle(obstacle.pos, obstacle.radius, color="gray"))
    for agent in world.agents:
        color = world.resource_colors[agent.resource]
        ax.add_artist(plt.Circle(agent.pos, agent.radius, color=color))
        log_poses = agent.get_log_poses()
        plt.plot(log_poses[0,:], log_poses[1,:], color=color)
        if agent.goal is not None:
            # Only plot active goals
            plt.scatter(agent.goal.pos[0], agent.goal.pos[1], color=color, marker="x", s=100)

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([corner1[0], corner2[0]])
    plt.ylim([corner2[1], corner1[1]])
    plt.show()

