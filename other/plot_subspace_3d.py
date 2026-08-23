import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

point  = np.array([1, 2, 3])
normal = np.array([1, 1, 2])

point2 = np.array([10, 50, 50])

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d = -point.dot(normal)

# create x,y
xx, yy = np.meshgrid(range(10), range(10))

# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

# create the figure
fig = plt.figure()

# add axes
ax = fig.add_subplot(111,projection='3d')

xx, yy = np.meshgrid(range(10), range(10))
z = 0.5 * (- 1 - np.sqrt(3)) * xx + (np.sqrt(3) - 1) * yy / (-3 + 2 * np.sqrt(3))

# plot the plane
ax.plot_surface(xx, yy, z, alpha=0.5)

n1 = 5 / 3
n2 = math.sqrt(5) / 3
a = 0.4

u1 = np.array([1., 1., 1.])
u1 /= np.linalg.norm(u1)
u2 = np.array([-1., 0., 1.])
u2 /= np.linalg.norm(u2)

soa = np.array([
    [4, 6, 6, *u1],
    [4, 6, 6, *u2],
    [4, 6, 6, -3 * a, -2 * a, 4 * a],
    [0, 0, 0, 4-3*a, 6-2*a, 6+4*a],
    [0, 0, 0, 4, 6, 6],
])
X, Y, Z, U, V, W = zip(*soa)
ax.quiver(
    X, Y, Z, U, V, W,
    color=('b', 'b', 'r', 'g', 'purple'),
)

ax.set_box_aspect([1,1,1])
plt.show()