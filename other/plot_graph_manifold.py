import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))

# =============
# First subplot
# =============
# set up the Axes for the first plot
ax = fig.add_subplot(1, 1, 1, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo
X = np.arange(-1.2, 1.2, 0.05)
Y = np.arange(-1.2, 1.2, 0.05)
X, Y = np.meshgrid(X, Y)
Z = np.exp(-(X**2 + Y**2))
# Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='brown',
                       linewidth=0, antialiased=False)
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
ax.set_zlim(-1.01, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()