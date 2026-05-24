import numpy as np

N = 10000


def sample_sphere(n):
    z = 1 - 2 * np.random.rand(n)
    phi = 2 * np.pi * np.random.rand(n)
    r = np.sqrt(1 - z**2)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack((x, y, z)), z


sphere_points, z_vals = sample_sphere(N)
