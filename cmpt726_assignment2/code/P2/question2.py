#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def y1(x1, x2):
    return 6 + 2 * x1 ** 2 + 2 * x2 ** 2


def y2(x1, x2):
    return 8 + 0 * x1 ** 2 + 0 * x2 ** 2

# Draw y1(x)
fig = plt.figure(1)
ax = plt.axes(projection='3d')
x1 = x2 = np.arange(-1.0, 1.0, 0.01)
(x1,x2)=np.meshgrid(x1,x2)
y1 = y1(x1,x2)
ax.plot_surface(x1, x2, y1, color='red')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y1')
plt.show()


# Draw y2(x)
fig = plt.figure(2)
ax = plt.axes(projection='3d')
x1 = x2 = np.arange(-1.0, 1.0, 0.01)
(x1,x2)=np.meshgrid(x1,x2)
y2 = y2(x1,x2)
ax.plot_surface(x1, x2, y2, color='blue')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y2')
plt.show()

# Draw decision boundary
fig=plt.figure(3)
plt.axis([-1,1,-1,1])
ax=fig.add_subplot(1,1,1)
circ=plt.Circle((0,0), radius=1, color='g', fill=False)
ax.add_patch(circ)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()

# Draw decision boundary
fig = plt.figure(4)
ax = plt.axes(projection='3d')
x1 = x2 = np.arange(-1.0, 1.0, 0.01)
(x1,x2)=np.meshgrid(x1,x2)
y2 = y2(x1,x2)
y1 = y1(x1,x2)
y3 = np.maximum(y1,y2)
ax.plot_surface(x1, x2, y1, color='blue')
ax.plot_surface(x1, x2, y2, color='green')
ax.plot_surface(x1, x2, y3, color='red')
plt.show()
