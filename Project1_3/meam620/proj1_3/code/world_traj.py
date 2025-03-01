import numpy as np
import math

from .graph_search import graph_search
from scipy.interpolate import make_interp_spline

def speed_map(dist, index, points):
    n = len(points)

    # Calculate the speed based on the distance
    # if dist < 1.0:
    #     curr_speed = 1.2
    # elif dist > 1.6:
    #     curr_speed = 3.7
    # elif dist > 2.0:
    #     curr_speed = 4.0
    # else:
    #     # Create a linear mapping.
    #     m = 1.1
    #     c = 1.2
    #     curr_speed = (c + m*(dist - 1)) #* 1.2

    # Calculate the speed based on the distance
    if dist < 1.0:
        curr_speed = 1.2
    elif dist > 3.0:
        curr_speed = 4.5
    elif dist > 1.8:
        curr_speed = 4.2
    else:
        # Create a linear mapping between 1.0 and 3.0
        curr_speed = (1.25 + (dist - 1.0) * (3.2 - 1.0) / 2.0) * 1.4

    #
    # # If prev_points is None or next_point is None
    if index == 0:
        curr_speed = curr_speed / 1.4
    #
    if index == n-2:
        curr_speed = curr_speed / 1.4
    #
    if index == n-3:
        curr_speed = curr_speed / 1.3


    # if index == n-4:
    #     curr_speed = curr_speed / 1.2

    # Calculate the angle between the current point and the next point
    if index != n-2:
        prev_point = points[index]
        curr_point = points[index+1]
        next_point = points[index+2]
        angle = math.acos(np.dot(curr_point - prev_point, next_point - curr_point) / (np.linalg.norm(curr_point - prev_point) * np.linalg.norm(next_point - curr_point)))
        if angle > np.radians(60):
            curr_speed = curr_speed/1.4

    # elif angle > np.radians(20)
    #     return curr_speed/1.1
    # elif angle > np.radians(20):
    #     return curr_speed/1.2
    # elif angle > np.radians(10):
    #     return curr_speed/1.1

    return curr_speed

def DouglasPeucker(points, epsilon):
    # Find the point with the maximum distance
    dmax = 0
    index = 0
    end = len(points)
    for i in range(1, end-1):
        d = np.linalg.norm(np.cross(points[i] - points[0], points[end-1] - points[0])) / np.linalg.norm(points[end-1] - points[0])
        if d > dmax:
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        rec_results1 = DouglasPeucker(points[:index+1], epsilon)
        rec_results2 = DouglasPeucker(points[index:], epsilon)

        # Build the result list
        results = np.vstack((rec_results1[:-1], rec_results2))
    else:
        results = np.vstack((points[0], points[-1]))

    return results


class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.2, 0.2, 0.2])
        self.margin = 0.6

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        # print(len(self.path))
        # print(self.path)
        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        # self.points = np.zeros((1,3)) # shape=(n_pts,3)
        self.points = DouglasPeucker(self.path, 0.19)
        # print(self.points)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE
        num_segments = len(self.points) - 1

        # Calculate the time taken to traverse each segment
        self.segment_times = np.zeros((num_segments,))
        for i in range(num_segments):
            dist = np.linalg.norm(self.points[i + 1] - self.points[i])
            self.segment_times[i] = dist / speed_map(dist, i, self.points)

        self.point_times = np.cumsum(self.segment_times)
        self.point_times = np.insert(self.point_times, 0, 0)


        self.spline = make_interp_spline(self.point_times, self.points, k=3, bc_type="clamped")
        # Calculate the total time taken to traverse the entire path
        self.total_time = np.sum(self.segment_times)

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        if t >= self.total_time:
            flat_output = {'x': self.points[-1], 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot,
                           'x_ddddot': x_ddddot,
                           'yaw': yaw, 'yaw_dot': yaw_dot}
            return flat_output

        x = self.spline(t)
        x_dot = self.spline(t, 1)
        x_ddot = self.spline(t, 2)
        x_dddot = self.spline(t, 3)
        x_ddddot = self.spline(t, 4)

        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}
        return flat_output
