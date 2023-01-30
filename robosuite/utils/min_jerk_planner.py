"""
Minimum jerk trajectory for 6DOF robot
Latest update: 14.12.2020
Written by Daniel Stankowski
"""
import numpy as np
import robosuite.utils.angle_transformation as at


def _calc_D(t0, tf):

    return np.array([
        [1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
        [0, 1, 2 * t0, 3 * (t0) ** 2, 4 * (t0) ** 3, 5 * (t0) ** 4],
        [0, 0, 2, 6 * (t0), 12 * (t0) ** 2, 20 * (t0) ** 3],
        [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
        [0, 1, 2 * tf, 3 * (tf) ** 2, 4 * (tf) ** 3, 5 * (tf) ** 4],
        [0, 0, 2, 6 * (tf), 12 * (tf) ** 2, 20 * (tf) ** 3]
    ])


def _calc_a(S, t0, tf):
    D = _calc_D(t0, tf)
    a = np.dot(np.linalg.inv(D), S)

    return a


def calc_kinematics(t, a):

    D = _calc_D(t, t)
    s = np.dot(D, a)

    return s


class PathPlan(object):
    """
    IMPORTANT: When the pose is passed [x,y,z,Rx,Ry,Rz] one has to convert the orientation
    part from axis-angle representation to the Euler before running this script.
    """

    def __init__(self, initial_point, target_point, total_time):
        self.initial_position = initial_point[:3]
        self.target_position = target_point[:3]
        self.initial_orientation = initial_point[3:]
        self.target_orientation = target_point[3:]
        self.tfinal = total_time
        self.desired_vec_fin = []

    def built_min_jerk_traj(self):
        """
        built minimum jerk (the desired trajectory)

        return:
        """
        pos = self.target_position
        ori = self.target_orientation
        pos_in = self.initial_position
        ori_in = self.initial_orientation

        self.X_fi = np.array([pos[0], 0, 0])
        self.Y_fi = np.array([pos[1], 0, 0])
        self.Z_fi = np.array([pos[2], 0, 0])

        self.X_t_fi = np.array([ori[0], 0, 0])
        self.Y_t_fi = np.array([ori[1], 0, 0])
        self.Z_t_fi = np.array([ori[2], 0, 0])

        X_in = np.array([pos_in[0], 0, 0])
        Y_in = np.array([pos_in[1], 0, 0])
        Z_in = np.array([pos_in[2], 0, 0])

        X_t_in = np.array([ori_in[0], 0, 0])
        Y_t_in = np.array([ori_in[1], 0, 0])
        Z_t_in = np.array([ori_in[2], 0, 0])

        S1_loc = np.array([X_in[0], 0, 0, self.X_fi[0], 0, 0])

        self.a1_loc = _calc_a(S1_loc, 0, self.tfinal)

        self.loc_x_y = (self.Y_fi - Y_in) / (self.X_fi - X_in + 1e-20)
        self.loc_x_z = (self.Z_fi - Z_in) / (self.X_fi - X_in + 1e-20)

        s1_angle = np.array([ori_in[0], 0, 0, self.X_t_fi[0], 0, 0])

        self.a1_angle = _calc_a(s1_angle, 0, self.tfinal)

        self.teta_x_y = (self.Y_t_fi - Y_t_in) / (self.X_t_fi - X_t_in + 1e-20)
        self.teta_x_z = (self.Z_t_fi - Z_t_in) / (self.X_t_fi - X_t_in + 1e-20)

    def trajectory_planning(self, t_now):

        kinem_x = calc_kinematics(t_now, self.a1_loc)[:3]

        kinem_y = self.loc_x_y * (kinem_x - self.X_fi) + self.Y_fi
        kinem_z = self.loc_x_z * (kinem_x - self.X_fi) + self.Z_fi

        kinem_teta_x = calc_kinematics(t_now, self.a1_angle)[:3]
        kinem_teta_y = self.teta_x_y * (kinem_teta_x - self.X_t_fi) + self.Y_t_fi
        kinem_teta_z = self.teta_x_z * (kinem_teta_x - self.X_t_fi) + self.Z_t_fi

        right_pos_desired = np.array([kinem_x[0], kinem_y[0], kinem_z[0]])
        right_ori_desired = np.array([kinem_teta_x[0], kinem_teta_y[0], kinem_teta_z[0]])

        right_vel_desired = np.array([kinem_x[1], kinem_y[1], kinem_z[1]])
        right_omega_desired = np.array([kinem_teta_x[1], kinem_teta_y[1], kinem_teta_z[1]])

        self.desired_vec_fin.append([right_pos_desired, right_ori_desired, right_vel_desired, right_omega_desired])

        return [right_pos_desired, right_ori_desired, right_vel_desired, right_omega_desired]
