# Use mplot 3d to plot the trajectory of the robot position
# Given by d(t) = [cos(0.1t), sin(0.12t), sin(0.08*t)].
# The pose of the robot is given by the rotation matrix R(t),
# and the position vector d(t).
# R(t) = [cos(t), -sin(t), 0]
#        [sin(t), cos(t), 0]
#        [0, 0, 1]
# On the robot, there is a fixed point at [c, 0, 0]. c = 0.25
# On the same 3d plot, plot the trajectory of the fixed point and the robot.
# Make sure to include a title, axis labels, and a legend.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def value_at_t(t):
    c = 0.25
    R = np.array([
        [np.cos(t), -np.sin(t), 0],
        [np.sin(t), np.cos(t), 0],
        [0, 0, 1]
    ])
    d = np.array([np.cos(0.1*t), np.sin(0.12*t), np.sin(0.08*t)])
    fixed_point = np.array([c, 0, 0])
    fixed_point_world_frame = np.matmul(R, fixed_point) + d
    return d, fixed_point_world_frame


def plot_trajectory():
    t = np.linspace(0, 100, 1000)
    c = 0.25
    d = []
    fixed_point = []
    for i in t:
        d_i, fixed_point_i = value_at_t(i)
        d.append(d_i)
        fixed_point.append(fixed_point_i)
    d = np.array(d).T
    fixed_point = np.array(fixed_point).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(d[0], d[1], d[2], label='Robot')
    ax.plot(fixed_point[0], fixed_point[1], fixed_point[2], label='Fixed Point')
    ax.set_title("Trajectory of Robot and Fixed Point")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend()
    plt.show()

plot_trajectory()