import matplotlib.pyplot as plt
import numpy as np

figure = plt.figure()
axes = figure.add_subplot(projection='3d')
# axes.set_xlim([-10,10])
# axes.set_ylim([-1,100])
# axes.set_zlim([-10,10])

matrix = np.loadtxt("output.txt")

x, y, z = matrix.T

axes.scatter(x, y, z)


axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_zlabel('z')

plt.show()