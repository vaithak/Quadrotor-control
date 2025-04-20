# %% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


def skew(v):
    """
    Function to compute the skew symmetric matrix of a vector
    :param v: 3x1 vector
    :return: 3x3 skew symmetric matrix
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


# %% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE

    new_q = q * Rotation.from_rotvec((w_m - w_b).squeeze() * dt)
    accel_world_frame = q.apply((a_m - a_b).squeeze()).reshape(3, 1) + g
    new_v = v + accel_world_frame * dt
    new_p = p + v * dt + 0.5 * accel_world_frame * (dt ** 2)

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    accel_unbiased = (a_m - a_b).squeeze()
    gyro_unbiased = (w_m - w_b).squeeze()
    q_conjugate = Rotation.from_rotvec(gyro_unbiased * dt).inv()
    accel_skew = skew(accel_unbiased)

    Fx = np.eye(18)
    Fx[0:3, 3:6] = np.eye(3) * dt
    Fx[3:6, 6:9] = (q.as_matrix() @ accel_skew) * dt * -1
    Fx[3:6, 9:12] = q.as_matrix() * dt * -1
    Fx[3:6, 15:18] = np.eye(3) * dt
    Fx[6:9, 6:9] = q_conjugate.as_matrix()
    Fx[6:9, 12:15] = -np.eye(3) * dt

    Fi = np.zeros((18, 12))
    Fi[3:6, 0:3] = np.eye(3)
    Fi[6:9, 3:6] = np.eye(3)
    Fi[9:12, 6:9] = np.eye(3)
    Fi[12:15, 9:12] = np.eye(3)

    Qi = np.zeros((12, 12))
    Qi[0:3, 0:3] = np.eye(3) * (accelerometer_noise_density * dt) ** 2
    Qi[3:6, 3:6] = np.eye(3) * (gyroscope_noise_density * dt) ** 2
    Qi[6:9, 6:9] = np.eye(3) * (accelerometer_random_walk) ** 2 * dt
    Qi[9:12, 9:12] = np.eye(3) * (gyroscope_random_walk) ** 2 * dt

    P_updated = Fx @ error_state_covariance @ Fx.T + Fi @ Qi @ Fi.T
    # return an 18x18 covariance matrix
    return P_updated


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    Pc = q.inv().apply((Pw - p).squeeze()).reshape(3, 1)
    Zc = Pc[2, 0]
    uv_hat = Pc[:2] / Zc
    uv_hat = uv_hat.reshape(2, 1)
    innovation = uv - uv_hat
    if norm(innovation) > error_threshold:
        return nominal_state, error_state_covariance, innovation
    H = np.zeros((2, 18))
    del_z_del_Pc = (1.0 / Zc) * np.array([[1, 0, -uv_hat[0, 0]],
                                          [0, 1, -uv_hat[1, 0]]])
    Pc_skew = skew(Pc.squeeze())
    H[0:2, 0:3] = -del_z_del_Pc @ q.inv().as_matrix()
    H[0:2, 6:9] = del_z_del_Pc @ Pc_skew

    K = error_state_covariance @ H.T @ inv(H @ error_state_covariance @ H.T + Q)
    delta_x = K @ innovation
    delta_x = delta_x.reshape(18, 1)
    p += delta_x[0:3]
    v += delta_x[3:6]
    q = q * Rotation.from_rotvec(delta_x[6:9].squeeze())
    a_b += delta_x[9:12]
    w_b += delta_x[12:15]
    g += delta_x[15:18]
    error_state_covariance = (np.eye(18) - K @ H) @ error_state_covariance @ (np.eye(18) - K @ H).T + K @ Q @ K.T
    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation
