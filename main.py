from world import World
import numpy as np

if __name__ == '__main__':
    N = 100
    T = 10.0
    dt = T / N
    world = World(2, 1000, N)
    world.add_planet(np.array([500, 500], dtype=np.float32), 100)
    world.add_planet(np.array([800, 500], dtype=np.float32), 80)
    world.add_agent(np.array([500, 300], dtype=np.float32), "A", "#ff0000")
    for i in range(N):
        world.update(dt, i)
    world.plot()
