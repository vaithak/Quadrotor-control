import numpy as np

class WaypointTraj(object):
    """

    """
    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """

        # STUDENT CODE HERE
        self.speed = 2.3 # m/s
        self.ts = np.array([np.linalg.norm(points[i+1]-points[i])/self.speed for i in range(len(points)-1)])
        self.total_time = np.sum(self.ts)
        self.points = points
                           

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
            flat_output = { 'x':self.points[-1], 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                            'yaw':yaw, 'yaw_dot':yaw_dot}
            return flat_output
        
        # Compute the current segment
        t_segment = 0
        idx = 0
        while idx < len(self.ts):
            if t < t_segment + self.ts[idx]:
                break
            t_segment += self.ts[idx]
            idx += 1
        if idx == len(self.ts):
            idx -= 1

        # Compute the current time in the segment
        t_segment = t - t_segment

        # Compute unit vector pointing to the next waypoint
        I_vec = np.zeros((3,))
        if idx < len(self.points)-1:
            I_vec = self.points[idx+1] - self.points[idx]
            I_vec = I_vec/np.linalg.norm(I_vec)

        # Compute the current position
        x = self.points[idx] + I_vec*self.speed*t_segment

        # Compute the current velocity
        x_dot = I_vec*self.speed

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
