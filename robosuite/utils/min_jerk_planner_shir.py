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

    def __init__(self, initial_point, target_point, total_time, via_point, time):
        self.initial_position = initial_point[:3]
        self.target_position = target_point[:3]
        self.initial_orientation = initial_point[3:]
        self.target_orientation = target_point[3:]
        self.tfinal = total_time
        self.t = time
        self.via_point = via_point
        self.desired_vec_fin = []

    def built_min_jerk_traj(self):
        """
        built minimum jerk (the desired trajectory)

        return:
        """
        pos = self.target_position
        ori = self.target_orientation
        # pos_in = self.initial_position

        if self.via_point == 0:
            pos_in = self.initial_position
            ori_in = self.initial_orientation
            ori_in[2] = abs(self.initial_orientation)
        else:
            if self.via_point == 1 and self.enter == 1:
                pos_in = self.desired_vec_fin[-1][0]
                ori_in = self.desired_vec_fin[-1][1]#T.mat2euler(self.sim.data.get_body_xmat("robot0_right_hand"))
        # if self.num_via_points > 2:
        #     if self.checked == 1:
        #         pos_in = pos  # self.peg_pos
        #         ori_in = ori  # T.mat2euler(self.sim.data.get_body_xmat("robot0_right_hand"))

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

        if self.via_point < (self.num_via_points-1):
            time_to = self.tfinal * (self.trans[str(self.via_point)])
        else:
            time_to = self.tfinal * (1 - self.trans[str(self.via_point-1)]) * 0.5
            # time_to = self.loop_time * (1 - sum(self.trans[str(x)] for x in range(self.num_via_points - 1)))  # * 0.1

        S1_loc = np.array([X_in[0], 0, 0, self.X_fi[0], 0, 0])

        self.a1_loc = _calc_a(S1_loc, 0, time_to)

        self.loc_x_y = (self.Y_fi - Y_in) / (self.X_fi - X_in + 1e-20)
        self.loc_x_z = (self.Z_fi - Z_in) / (self.X_fi - X_in + 1e-20)

        s1_angle = np.array([ori_in[0], 0, 0, self.X_t_fi[0], 0, 0])

        self.a1_angle = _calc_a(s1_angle, 0, time_to)

        self.teta_x_y = (self.Y_t_fi - Y_t_in) / (self.X_t_fi - X_t_in + 1e-20)
        self.teta_x_z = (self.Z_t_fi - Z_t_in) / (self.X_t_fi - X_t_in + 1e-20)

    def trajectory_planning(self):
        if self.via_point == 1 and self.enter == 1:
            self.t_bias = self.t - 0.002
            self.enter = 0
        t_now = self.t - self.t_bias

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

    def calc_next_desired_point(self):
        """"
        create minimum jerk trajectory.
        calculate the next desired point in the desired trajectory.
        first, built minimum jerk trajectory, then, built the next desired point,
        taking in consideration hhe current time and calculate it.
        base on: https://www.researchgate.net/publication/329954197_A_Novel_Tuning_Method_for_PD_Control_of_Robotic_Manipulators_Based_on_Minimum_Jerk_Principle

        return: vec of desired pos, ori, vel and ori_dot (size 1,12)


        """
        if self.via_point < (self.num_via_points-1):
            if self.switch == 3 or self.timestep == round((self.horizon * self.trans[str(self.checked)])):
                self.via_point += 1
                self.built_min_jerk_traj()
            else:
                if self.switch > 3:
                    self.switch = 0
                    self.switch_seq = 0

        if self.t == 0.:
            self.built_min_jerk_traj()

        return self.built_next_desired_point()
