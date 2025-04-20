import numpy as np
from scipy.spatial.transform import Rotation


class SE3Control(object):
    """

    """

    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass = quad_params['mass']  # kg
        self.Ixx = quad_params['Ixx']  # kg*m^2
        self.Iyy = quad_params['Iyy']  # kg*m^2
        self.Izz = quad_params['Izz']  # kg*m^2
        self.arm_length = quad_params['arm_length']  # meters
        self.rotor_speed_min = quad_params['rotor_speed_min']  # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max']  # rad/s
        self.k_thrust = quad_params['k_thrust']  # N/(rad/s)**2
        self.k_drag = quad_params['k_drag']  # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.g = 9.81  # m/s^2

        # STUDENT CODE HERE
        # self.kp_x = np.diag([3.8, 3.8, 3.8])
        # self.kd_x = np.diag([4.3, 4.3, 3.75])
        self.kp_x = np.diag([4.0, 4.0, 3.58])
        self.kd_x = np.diag([3.5, 3.5, 3.05])

        self.k_R = np.diag([100, 100, 100])
        self.k_w = np.diag([10, 10, 10])

        gamma = self.k_drag / self.k_thrust
        self.F_to_u_matrix = np.array([[1, 1, 1, 1],
                                       [0, self.arm_length, 0, -self.arm_length],
                                       [-self.arm_length, 0, self.arm_length, 0],
                                       [gamma, -gamma, gamma, -gamma]])
        self.u_to_F_matrix = np.linalg.inv(self.F_to_u_matrix)

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        acceleration_req = flat_output['x_ddot'] - self.kd_x @ (state['v'] - flat_output['x_dot']) - self.kp_x @ (
                    state['x'] - flat_output['x'])

        F_desired = self.mass * acceleration_req + np.array([0, 0, self.mass * self.g])
        R_current = Rotation.from_quat(state['q']).as_matrix()
        cmd_thrust = F_desired.T @ R_current[:, 2]

        b3 = F_desired / np.linalg.norm(F_desired)
        a_yaw = np.array([np.cos(flat_output['yaw']), np.sin(flat_output['yaw']), 0])
        b2 = np.cross(b3, a_yaw) / np.linalg.norm(np.cross(b3, a_yaw))
        b1 = np.cross(b2, b3)
        R_desired = np.array([b1, b2, b3]).T
        cmd_q = Rotation.from_matrix(R_desired).as_quat()

        R_rel = np.dot(R_desired.T, R_current)
        e_R = 0.5 * np.array([R_rel[2, 1] - R_rel[1, 2], R_rel[0, 2] - R_rel[2, 0], R_rel[1, 0] - R_rel[0, 1]])
        e_w = state['w'] - np.array([0, 0, flat_output['yaw_dot']])

        cmd_moment = np.dot(self.inertia, (-np.dot(self.k_R, e_R) - np.dot(self.k_w, e_w)))

        # Calculate motor speeds
        u = np.concatenate(([cmd_thrust], cmd_moment))
        F = self.u_to_F_matrix @ u
        cmd_motor_speeds = np.sqrt(np.clip(F / self.k_thrust, self.rotor_speed_min**2, self.rotor_speed_max**2))

        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment,
                         'cmd_q': cmd_q}
        return control_input