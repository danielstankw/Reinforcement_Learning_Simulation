import pickle
import timeit
import random
import sklearn
from robosuite.controllers.base_controller import Controller
import robosuite.utils.transform_utils as T
import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from copy import deepcopy

# Supported impedance modes
ERROR_TOP = 0.0007
PEG_RADIUS = 0.0021
HOLE_RADIUS = 0.0024

class ImpedanceSpiralMeasurements(Controller):
    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 input_max=1,
                 input_min=-1,
                 output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
                 output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
                 damping_ratio=1,
                 impedance_mode="fixed",
                 kp_limits=(0, 300),
                 damping_ratio_limits=(0, 100),
                 policy_freq=20,
                 position_limits=None,
                 orientation_limits=None,
                 interpolator_pos=None,
                 interpolator_ori=None,
                 control_ori=True,
                 control_delta=True,
                 uncouple_pos_ori=True,
                 control_dim=36,
                 plotter=False,
                 ori_method='rotation',
                 show_params=True,
                 total_time=0,
                 circle=True,

                 **kwargs  # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )
        self.save_ee_pos_x = None
        self.save_ee_pos_y = None
        self.save_ee_pos_z = None
        self.save_ee_rot_x = None
        self.save_ee_rot_y = None
        self.save_ee_rot_z = None
        self.save_lin_vel_x = None
        self.save_lin_vel_y = None
        self.save_lin_vel_z = None
        self.save_fx = None
        self.save_fy = None
        self.save_fz = None
        self.save_mx = None
        self.save_my = None
        self.save_mz = None
        self.save_case = None
        self.save_t_contact = None
        self.save_time = None
        self.stop = False

        self.peg_pos_z = None
        self.radius_next = 0
        self.x_current = 0
        self.y_current = 0
        self.overlap = False

        self.circle = circle

        # spiral parameters
        self.spiral_flag = True
        self.x_spiral_next = 0
        self.y_spiral_next = 0
        self.theta_current = 0
        self.radius_current = 0

        self.end_wait = 0
        self.initialContactTime = 0
        self.existsOverlap = False
        self.overlap_time = None

        self.insertion = False

        # for plotting:
        self.total_time = total_time
        self.show_params = show_params
        self.plotter = plotter
        self.method = ori_method
        self.PartialImpedance = False
        # Determine whether this is pos ori or just pos
        self.use_ori = control_ori

        # Determine whether we want to use delta or absolute values as inputs
        self.use_delta = control_delta

        self.control_dim = control_dim

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # kp and kd limits
        self.kp_min = self.nums2array(kp_limits[0], 6)
        self.kp_max = self.nums2array(kp_limits[1], 6)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], 6)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], 6)

        self.kp = np.array([250000.0, 250000.0, 700.0, 700.0, 700.0, 700.0])
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio

        self.kp_impedance = deepcopy(self.kp)
        self.kd_impedance = deepcopy(self.kd)

        # Impedance mode
        self.impedance_mode = impedance_mode

        # limits
        self.position_limits = position_limits
        self.orientation_limits = orientation_limits

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator_pos = interpolator_pos
        self.interpolator_ori = interpolator_ori

        # whether pos and ori want to be uncoupled
        self.uncoupling = uncouple_pos_ori

        # initialize goals based on initial pos / ori
        self.goal_ori = np.array(self.initial_ee_ori_mat)
        self.goal_pos = np.array(self.initial_ee_pos)
        self.goal_vel = np.array(self.ee_pos_vel)
        self.goal_ori_vel = np.array(self.initial_ee_ori_vel)
        self.impedance_vec = np.zeros(12)
        self.switch = 0
        self.madeContact = False
        self.force_filter = np.zeros((3, 1))

        self.relative_ori = np.zeros(3)
        self.ori_ref = None
        self.set_desired_goal = False
        self.desired_pos = np.zeros(12)
        self.torques = np.zeros(6)
        self.F0 = np.zeros(6)
        self.F_int = np.zeros(6)
        self.measured_sensor_bias = False
        # ee resets - bias at initial state
        self.ee_sensor_bias = 0

        # for graphs

        self.time_vec = []
        # robot measurements
        self.ee_pos_x_vec, self.ee_pos_y_vec, self.ee_pos_z_vec = [], [], []
        self.ee_vel_x_vec, self.ee_vel_y_vec, self.ee_vel_z_vec = [], [], []
        self.ee_ori_x_vec, self.ee_ori_y_vec, self.ee_ori_z_vec = [], [], []
        self.ee_ori_vel_x_vec, self.ee_ori_vel_y_vec, self.ee_ori_vel_z_vec = [], [], []
        # minimum jerk
        self.pos_min_jerk_x, self.pos_min_jerk_y, self.pos_min_jerk_z = [], [], []
        self.vel_min_jerk_x, self.vel_min_jerk_y, self.vel_min_jerk_z = [], [], []
        self.ori_min_jerk_x, self.ori_min_jerk_y, self.ori_min_jerk_z = [], [], []
        self.ori_vel_min_jerk_x, self.ori_vel_min_jerk_y, self.ori_vel_min_jerk_z = [], [], []
        # impedance
        self.impedance_pos_vec_x, self.impedance_pos_vec_y, self.impedance_pos_vec_z = [], [], []
        self.impedance_ori_vec_x, self.impedance_ori_vec_y, self.impedance_ori_vec_z = [], [], []
        self.impedance_vel_vec_x, self.impedance_vel_vec_y, self.impedance_vel_vec_z = [], [], []
        self.impedance_ori_vel_vec_x, self.impedance_ori_vel_vec_y, self.impedance_ori_vel_vec_z = [], [], []
        # wrench - based on PD
        self.applied_wrench_fx, self.applied_wrench_fy, self.applied_wrench_fz = [], [], []
        self.applied_wrench_mx, self.applied_wrench_my, self.applied_wrench_mz = [], [], []
        # sensor readings
        self.sensor_fx, self.sensor_fy, self.sensor_fz = [], [], []
        self.sensor_mx, self.sensor_my, self.sensor_mz = [], [], []
        # spiral
        self.spiral_x = []
        self.spiral_y = []
        self.robot_spiral_x = []
        self.robot_spiral_y = []
        self.zones = []

    def set_goal(self):
        """
        Pre-plans minimum jerk trajectories:
        a) first one from initial point (switch=0) -> point above hole (switch=1)
        b) second: from above the hole -> inside the hole
        """
        # Update state
        self.update()

        if self.switch == 0:
            self.goal_pos, self.goal_ori, self.goal_vel, self.goal_ori_vel = self.built_next_desired_point()
        else:  # if switch == 1 i.e. above the hole
            if self.sim.data.time - self.t_bias < self.t_finial:
                self.goal_pos, self.goal_ori, self.goal_vel, self.goal_ori_vel = self.built_next_desired_point()
            else:
                self.goal_pos, self.goal_ori, self.goal_vel, self.goal_ori_vel = self.desired_vec_fin[-1]

        self.set_desired_goal = True

    def run_controller(self):
        """
        Calculates the torques required to reach the desired set point.
        Impedance Position Base (IM-PB) -- position and orientation.
        work in world space
        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()

        self.desired_pos = np.concatenate((self.goal_pos, self.goal_ori, self.goal_vel, self.goal_ori_vel), axis=0)

        if self.switch and self.measured_sensor_bias is False:
            """We take sensor bias measurement once when we reach to via_point=0 i.e. switch = 1
                this bias is used to calculate sensor measurements with less noise"""
            self.ee_sensor_bias = deepcopy(np.concatenate(
                (self.ee_ori_mat @ -self.sim.data.sensordata[:3], self.ee_ori_mat @ -self.sim.data.sensordata[3:]),
                axis=0))
            self.measured_sensor_bias = True # from False

        # sensor measurements
        self.F_int = (np.concatenate(
            (self.ee_ori_mat @ -self.sim.data.sensordata[:3], self.ee_ori_mat @ -self.sim.data.sensordata[3:]),
            axis=0) - self.ee_sensor_bias)

        if self.find_contacts() and self.madeContact is False:
            """
            Check if/when the initial contact with the hole surface has been established
                self.madeContact: [True/False] if contact with board was made
                self.initialContactTime: time of initial contact with board (for plotting)
            """
            self.madeContact = True
            self.initialContactTime = self.sim.data.time
            self.peg_pos_z = self.sim.data.get_site_xpos("peg_site")[2]
            print('%%%%%%%%%% Hole contact established %%%%%%%%%%%')
            print('Initializing contact PD parameters at the time of contact')
            # self.pxy = random.uniform(15000, 250000)
            # self.pz = random.uniform(50, 1000)
            # self.kp = np.array([self.pxy, self.pxy, self.pz, 450.0, 450.0, 450.0])
            self.kp = np.array([250000.0, 250000.0, 250.0, 450.0, 450.0, 450.0])
            self.kd = 2 * np.sqrt(self.kp) * np.sqrt(2)
            # print('kp contact', self.kp)
            # print('kd contact', self.kp)

        # for label collection
        case = self.zone_checker()
        if self.madeContact:
            self.zones.append(case)

        if self.circle_check() and self.madeContact and self.overlap_time is None and self.existsOverlap is False:
            """
            Used under assumption that correct hole position is known!
            IF:
                self.circle_check(): checks if peg is within radial distance from true hole specified by ERROR_TOP
            THEN:
                self.overlap_time: collect time of geometric overlap
                self.existsOverlap: [True/False] geometric overlap
                self.spiral_flag: [True/False] whether to stop [False] or continue spiral once overlap has been detected
            """
            self.overlap_time = self.sim.data.time
            self.existsOverlap = False

        # we don't want to do spiral search in the hole, so if peg tip
        # goes below surface level of hole than we turn off spiral search
        if self.madeContact:
            """If contact was made and we use spiral search -> update the trajectory"""
            if self.circle:
                """Circle mode"""
                theta_next, radius_next, self.x_spiral_next, self.y_spiral_next = self.next_circle(
                    self.theta_current)

                self.spiral_x.append(self.x_spiral_next + self.desired_pos[0])
                self.spiral_y.append(self.y_spiral_next + self.desired_pos[1])

            else:
                """Spiral Search mode"""
                theta_next, radius_next, self.x_spiral_next, self.y_spiral_next = \
                    self.next_spiral(self.theta_current)
                # add shift to the spiral search which is planned at (0,0)
                self.spiral_x.append(self.x_spiral_next + self.desired_pos[0])
                self.spiral_y.append(self.y_spiral_next + self.desired_pos[1])

            self.theta_current = deepcopy(theta_next)
            self.radius_current = deepcopy(radius_next)

            # we collect spiral trajectory at this point to exclude everything before contact was made
            self.robot_spiral_x.append(self.ee_pos[0])
            self.robot_spiral_y.append(self.ee_pos[1])
            # print('Spiral')

        self.desired_pos[:2] += np.array([self.x_spiral_next, self.y_spiral_next])
        ori_real = T.Rotation_Matrix_To_Vector(self.final_orientation, self.ee_ori_mat)

        # print('------------------------------')
        # error calculation
        ori_error = self.desired_pos[3:6] - ori_real
        vel_ori_error = self.desired_pos[9:12] - self.ee_ori_vel
        position_error = self.desired_pos[:3].T - self.ee_pos
        vel_pos_error = self.desired_pos[6:9].T - self.ee_pos_vel

        # Compute desired force and torque based on errors
        desired_force = (np.multiply(np.array(position_error), np.array(self.kp[0:3]))
                         + np.multiply(vel_pos_error, self.kd[0:3]))

        desired_torque = (np.multiply(np.array(ori_error), np.array(self.kp[3:6]))
                          + np.multiply(vel_ori_error, self.kd[3:6]))
        if self.madeContact:
            desired_force[2] = -5
        decoupled_wrench = np.concatenate([desired_force, desired_torque])
        self.torques = np.dot(self.J_full.T, decoupled_wrench).reshape(6, ) + self.torque_compensation

        self.set_desired_goal = False
        # Always run superclass call for any cleanups at the end
        super().run_controller()
        if np.isnan(self.torques).any():
            self.torques = np.zeros(6)


        if self.madeContact:

            # for graphs:

            # real_forces = np.dot(np.linalg.inv(self.J_full.T), self.sim.data.qfrc_actuator[:6]).reshape(6, )
            self.time_vec.append(self.sim.data.time)
            # robot measurements
            self.ee_pos_x_vec.append(self.ee_pos[0])
            self.ee_pos_y_vec.append(self.ee_pos[1])
            self.ee_pos_z_vec.append(self.ee_pos[2])
            self.ee_vel_x_vec.append((self.ee_pos_vel[0]))
            self.ee_vel_y_vec.append((self.ee_pos_vel[1]))
            self.ee_vel_z_vec.append((self.ee_pos_vel[2]))
            self.ee_ori_x_vec.append(ori_real[0])
            self.ee_ori_y_vec.append(ori_real[1])
            self.ee_ori_z_vec.append(ori_real[2])
            self.ee_ori_vel_x_vec.append(self.ee_ori_vel[0])
            self.ee_ori_vel_y_vec.append(self.ee_ori_vel[1])
            self.ee_ori_vel_z_vec.append(self.ee_ori_vel[2])
            # minimum jerk
            self.pos_min_jerk_x.append(self.goal_pos[0])
            self.pos_min_jerk_y.append(self.goal_pos[1])
            self.pos_min_jerk_z.append(self.goal_pos[2])
            self.vel_min_jerk_x.append(self.goal_vel[0])
            self.vel_min_jerk_y.append(self.goal_vel[1])
            self.vel_min_jerk_z.append(self.goal_vel[2])
            self.ori_min_jerk_x.append(self.goal_ori[0])
            self.ori_min_jerk_y.append(self.goal_ori[1])
            self.ori_min_jerk_z.append(self.goal_ori[2])
            self.ori_vel_min_jerk_x.append(self.goal_ori_vel[0])
            self.ori_vel_min_jerk_y.append(self.goal_ori_vel[1])
            self.ori_vel_min_jerk_z.append(self.goal_ori_vel[2])
            # impedance
            self.impedance_pos_vec_x.append(self.desired_pos[0])
            self.impedance_pos_vec_y.append(self.desired_pos[1])
            self.impedance_pos_vec_z.append(self.desired_pos[2])
            self.impedance_ori_vec_x.append(self.desired_pos[3])
            self.impedance_ori_vec_y.append(self.desired_pos[4])
            self.impedance_ori_vec_z.append(self.desired_pos[5])
            self.impedance_vel_vec_x.append(self.desired_pos[6])
            self.impedance_vel_vec_y.append(self.desired_pos[7])
            self.impedance_vel_vec_z.append(self.desired_pos[8])
            self.impedance_ori_vel_vec_x.append(self.desired_pos[9])
            self.impedance_ori_vel_vec_y.append(self.desired_pos[10])
            self.impedance_ori_vel_vec_z.append(self.desired_pos[11])
            # wrench - based on PD
            self.applied_wrench_fx.append(decoupled_wrench[0])
            self.applied_wrench_fy.append(decoupled_wrench[1])
            self.applied_wrench_fz.append(decoupled_wrench[2])
            self.applied_wrench_mx.append(decoupled_wrench[3])
            self.applied_wrench_my.append(decoupled_wrench[4])
            self.applied_wrench_mz.append(decoupled_wrench[5])
            # sensor readings
            self.sensor_fx.append(self.F_int[0])
            self.sensor_fy.append(self.F_int[1])
            self.sensor_fz.append(self.F_int[2])
            self.sensor_mx.append(self.F_int[3])
            self.sensor_my.append(self.F_int[4])
            self.sensor_mz.append(self.F_int[5])

        return self.torques

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property
        Returns the following (generalized for both high and low limits), based on the impedance mode:
            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]
        Returns:
            2-tuple:
                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        if self.impedance_mode == "variable":
            low = np.concatenate([self.damping_ratio_min, self.kp_min, self.input_min])
            high = np.concatenate([self.damping_ratio_max, self.kp_max, self.input_max])
        elif self.impedance_mode == "variable_kp":
            low = np.concatenate([self.kp_min, self.input_min])
            high = np.concatenate([self.kp_max, self.input_max])
        else:  # This is case "fixed"
            low, high = self.input_min, self.input_max
        return low, high

    @property
    def name(self):
        return 'IMPEDANCE_PB_Partial'

    def ImpedanceEq(self, F_int, F0, x0, th0, x0_d, th0_d, dt):
        """
        Impedance Eq: F_int-F0=K(x0-xm)+C(x0_d-xm_d)-Mxm_dd
        Solving the impedance equation for x(k+1)=Ax(k)+Bu(k) where
        x(k+1)=[Xm,thm,Xm_d,thm_d]
        Parameters:
            x0,x0_d,th0,th0_d - desired goal position/orientation and velocity
            F_int - measured force/moments in [N/Nm] (what the robot sense)
            F0 - desired applied force/moments (what the robot does)
            xm_pose - impedance model (updated in a loop) initialized at the initial pose of robot
            A_d, B_d - A and B matrices of x(k+1)=Ax(k)+Bu(k)
        Output:
            X_nex = x(k+1) = [Xm,thm,Xm_d,thm_d]
        """

        # state space formulation
        # X=[xm;thm;xm_d;thm_d] U=[F_int;M_int;x0;th0;x0d;th0d]
        A_1 = np.concatenate((np.zeros([6, 6], dtype=int), np.identity(6)), axis=1)
        A_2 = np.concatenate((np.dot(-np.linalg.pinv(self.M), self.K), np.dot(-np.linalg.pinv(self.M), self.C)), axis=1)
        A_temp = np.concatenate((A_1, A_2), axis=0)

        B_1 = np.zeros([6, 18], dtype=int)
        B_2 = np.concatenate((np.linalg.pinv(self.M), np.dot(np.linalg.pinv(self.M), self.K),
                              np.dot(np.linalg.pinv(self.M), self.C)), axis=1)
        B_temp = np.concatenate((B_1, B_2), axis=0)

        if np.isnan(A_temp).any() or np.isnan(B_temp).any():
            s = 1
        # discrete state space A, B matrices
        A_d = expm(A_temp * dt)
        B_d = np.dot(np.dot(np.linalg.pinv(A_temp), (A_d - np.identity(A_d.shape[0]))), B_temp)

        # impedance model xm is initialized to initial position of the EEF and modified by force feedback
        xm = self.impedance_vec[:3].reshape(3, 1)
        thm = self.impedance_vec[3:6].reshape(3, 1)
        xm_d = self.impedance_vec[6:9].reshape(3, 1)
        thm_d = self.impedance_vec[9:12].reshape(3, 1)

        # State Space vectors
        X = np.concatenate((xm, thm, xm_d, thm_d), axis=0)  # 12x1 column vector

        U = np.concatenate((-F0 + F_int, x0, th0, x0_d, th0_d), axis=0).reshape(18, 1)

        # discrete state solution X(k+1)=Ad*X(k)+Bd*U(k)
        X_nex = np.dot(A_d, X) + np.dot(B_d, U)
        # print(X_nex[9:12])
        self.impedance_vec = deepcopy(X_nex)
        return X_nex.reshape(12, )

    def set_control_param(self, action):

        if self.control_dim == 36:
            self.K = np.array([[action[0], 0, 0, 0, action[1], 0],
                               [0, action[2], 0, action[3], 0, 0],
                               [0, 0, action[4], 0, 0, 0],
                               [0, action[5], 0, action[6], 0, 0],
                               [action[7], 0, 0, 0, action[8], 0],
                               [0, 0, 0, 0, 0, action[9]]])

            self.C = np.array([[action[10], 0, 0, 0, action[11], 0],
                               [0, action[12], 0, action[13], 0, 0],
                               [0, 0, action[14], 0, 0, 0],
                               [0, action[15], 0, action[16], 0, 0],
                               [action[17], 0, 0, 0, action[18], 0],
                               [0, 0, 0, 0, 0, action[19]]])

            self.M = np.array([[action[20], 0, 0, 0, action[21], 0],
                               [0, action[22], 0, action[23], 0, 0],
                               [0, 0, action[24], 0, 0, 0],
                               [0, action[25], 0, action[26], 0, 0],
                               [action[27], 0, 0, 0, action[28], 0],
                               [0, 0, 0, 0, 0, action[29]]])
            self.kp_impedance = action[-6:]

        if self.control_dim == 24:
            self.K = np.array([[action[0], 0, 0, 0, 0, 0],
                               [0, action[1], 0, 0, 0, 0],
                               [0, 0, action[2], 0, 0, 0],
                               [0, 0, 0, action[3], 0, 0],
                               [0, 0, 0, 0, action[4], 0],
                               [0, 0, 0, 0, 0, action[5]]])

            self.C = np.array([[action[6], 0, 0, 0, 0, 0],
                               [0, action[7], 0, 0, 0, 0],
                               [0, 0, action[8], 0, 0, 0],
                               [0, 0, 0, action[9], 0, 0],
                               [0, 0, 0, 0, action[10], 0],
                               [0, 0, 0, 0, 0, action[11]]])

            self.M = np.array([[action[12], 0, 0, 0, 0, 0],
                               [0, action[13], 0, 0, 0, 0],
                               [0, 0, action[14], 0, 0, 0],
                               [0, 0, 0, action[15], 0, 0],
                               [0, 0, 0, 0, action[16], 0],
                               [0, 0, 0, 0, 0, action[17]]])

            self.kp_impedance = action[-6:]
            self.kd_impedance = 2 * np.sqrt(self.kp_impedance) * np.sqrt(2)
        if self.control_dim == 26:
            self.K = np.array([[abs(action[0]), 0, 0, 0, action[1], 0],
                               [0, abs(action[2]), 0, action[3], 0, 0],
                               [0, 0, abs(action[4]), 0, 0, 0],
                               [0, action[5], 0, abs(action[6]), 0, 0],
                               [action[7], 0, 0, 0, abs(action[8]), 0],
                               [0, 0, 0, 0, 0, abs(action[9])]])

            self.C = np.array([[abs(action[10]), 0, 0, 0, action[11], 0],
                               [0, abs(action[12]), 0, action[13], 0, 0],
                               [0, 0, abs(action[14]), 0, 0, 0],
                               [0, action[15], 0, abs(action[16]), 0, 0],
                               [action[17], 0, 0, 0, abs(action[18]), 0],
                               [0, 0, 0, 0, 0, abs(action[19])]])

            self.M = np.array([[abs(action[20]), 0, 0, 0, 0, 0],
                               [0, abs(action[21]), 0, 0, 0, 0],
                               [0, 0, abs(action[22]), 0, 0, 0],
                               [0, 0, 0, abs(action[23]), 0, 0],
                               [0, 0, 0, 0, abs(action[24]), 0],
                               [0, 0, 0, 0, 0, abs(action[25])]])

            # self.K = np.array([[24.51158142, 0., 0., 0., -42.63611603, 0.],
            #                    [0., 40.25193405, 0., 26.59643364, 0., 0.],
            #                    [0., 0., 23.69382477, 0., 0., 0.],
            #                    [0., -10.66889191, 0., 3.27396274, 0., 0.],
            #                    [-19.22241402, 0., 0., 0., 46.74688339, 0.],
            #                    [0., 0., 0., 0., 0., 24.80905724]])
            # self.C = np.array([[66.41168976, 0., 0., 0., -26.20734787, 0.],
            #                    [0., 98.06533051, 0., 26.14341736, 0., 0.],
            #                    [0., 0., 103.66620636, 0., 0., 0.],
            #                    [0., 21.57993698, 0., 55.37984467, 0., 0.],
            #                    [-0.29122657, 0., 0., 0., 0.14158408, 0.],
            #                    [0., 0., 0., 0., 0., 2.37160134]])
            # self.M = np.array([[112.29223633, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                     [0.0, 72.80897522, 0.0, 0.0, 0.0, 0.0],
            #                     [0.0, 0.0, 169.45898438, 0.0, 0.0, 0.0],
            #                     [0.0, 0.0, 0.0, 37.9505806, 0.0, 0.0],
            #                     [0.0, 0.0, 0.0, 0.0, 4.87572193, 0.0],
            #                     [0.0, 0.0, 0.0, 0.0, 0.0, 14.63672161]])
            # print(self.K)
            # print(self.C)
            # print(self.M)
            # self.C = np.nan_to_num(2 * np.sqrt(np.dot(self.K, self.M)))
            # self.kp_impedance = np.array([700., 500., 100., 450., 450., 450.])
            # self.kd_impedance = 2 * np.sqrt(self.kp_impedance) * np.sqrt(2)
            # self.kd_impedance[3:] = 30

        if self.control_dim == 32:
            self.K = np.array([[abs(action[0]), 0, 0, 0, action[1], 0],
                               [0, abs(action[2]), 0, action[3], 0, 0],
                               [0, 0, abs(action[4]), 0, 0, 0],
                               [0, action[5], 0, abs(action[6]), 0, 0],
                               [action[7], 0, 0, 0, abs(action[8]), 0],
                               [0, 0, 0, 0, 0, abs(action[9])]])

            self.C = np.array([[abs(action[10]), 0, 0, 0, action[11], 0],
                               [0, abs(action[12]), 0, action[13], 0, 0],
                               [0, 0, abs(action[14]), 0, 0, 0],
                               [0, action[15], 0, abs(action[16]), 0, 0],
                               [action[17], 0, 0, 0, abs(action[18]), 0],
                               [0, 0, 0, 0, 0, abs(action[19])]])

            self.M = np.array([[abs(action[20]), 0, 0, 0, 0, 0],
                               [0, abs(action[21]), 0, 0, 0, 0],
                               [0, 0, abs(action[22]), 0, 0, 0],
                               [0, 0, 0, abs(action[23]), 0, 0],
                               [0, 0, 0, 0, abs(action[24]), 0],
                               [0, 0, 0, 0, 0, abs(action[25])]])

            self.kp_impedance = np.abs(action[26:32])
            self.kd_impedance = 2 * np.sqrt(self.kp_impedance) * np.sqrt(2)

        if self.control_dim == 38:
            self.K = np.array([[abs(action[0]), 0, 0, 0, action[1], 0],
                               [0, abs(action[2]), 0, action[3], 0, 0],
                               [0, 0, abs(action[4]), 0, 0, 0],
                               [0, action[5], 0, abs(action[6]), 0, 0],
                               [action[7], 0, 0, 0, abs(action[8]), 0],
                               [0, 0, 0, 0, 0, abs(action[9])]])

            self.C = np.array([[abs(action[10]), 0, 0, 0, action[11], 0],
                               [0, abs(action[12]), 0, action[13], 0, 0],
                               [0, 0, abs(action[14]), 0, 0, 0],
                               [0, action[15], 0, abs(action[16]), 0, 0],
                               [action[17], 0, 0, 0, abs(action[18]), 0],
                               [0, 0, 0, 0, 0, abs(action[19])]])

            self.M = np.array([[abs(action[20]), 0, 0, 0, 0, 0],
                               [0, abs(action[21]), 0, 0, 0, 0],
                               [0, 0, abs(action[22]), 0, 0, 0],
                               [0, 0, 0, abs(action[23]), 0, 0],
                               [0, 0, 0, 0, abs(action[24]), 0],
                               [0, 0, 0, 0, 0, abs(action[25])]])

            self.kp_impedance = np.abs(action[26:32])
            self.kd_impedance = np.abs(action[32:38])

            print('-------------------------K--------------------------------')
            print(self.K)
            print('-------------------------C--------------------------------')
            print(self.C)
            print('-------------------------M--------------------------------')
            print(self.M)
            print('-------------------------kp--------------------------------')
            print(self.kp_impedance)
            print('-------------------------kd--------------------------------')
            print(self.kd_impedance)

        if self.control_dim == 18:
            self.K = np.array([[abs(action[0]), 0, 0, 0, 0, 0],
                               [0, abs(action[1]), 0, 0, 0, 0],
                               [0, 0, abs(action[2]), 0, 0, 0],
                               [0, 0, 0, abs(action[3]), 0, 0],
                               [0, 0, 0, 0, abs(action[4]), 0],
                               [0, 0, 0, 0, 0, abs(action[5])]])

            self.C = np.array([[abs(action[6]), 0, 0, 0, 0, 0],
                               [0, abs(action[7]), 0, 0, 0, 0],
                               [0, 0, abs(action[8]), 0, 0, 0],
                               [0, 0, 0, abs(action[9]), 0, 0],
                               [0, 0, 0, 0, abs(action[10]), 0],
                               [0, 0, 0, 0, 0, abs(action[11])]])

            self.M = np.array([[abs(action[12]), 0, 0, 0, 0, 0],
                               [0, abs(action[13]), 0, 0, 0, 0],
                               [0, 0, abs(action[14]), 0, 0, 0],
                               [0, 0, 0, abs(action[15]), 0, 0],
                               [0, 0, 0, 0, abs(action[16]), 0],
                               [0, 0, 0, 0, 0, abs(action[17])]])

            self.kp_impedance = np.array([700., 500., 100., 450., 450., 450.])
            self.kd_impedance = 2 * np.sqrt(self.kp_impedance) * np.sqrt(2)
            self.kd_impedance[3:] = 30

            if self.show_params:
                print('-------------------------K--------------------------------')
                print(self.K)
                print('-------------------------C--------------------------------')
                print(self.C)
                print('-------------------------M--------------------------------')
                print(self.M)

    def find_contacts(self):
        gripper_geom_id = self.sim.model._geom_name2id['peg_g0']
        hole_geom_id = list(range(59, 72))
        hole_geom_id.append(7)
        hole_geom_id.append(8)
        if self.sim.data.ncon > 1:
            for i in range(self.sim.data.ncon):
                contact = self.sim.data.contact[i]
                if ((contact.geom1 == gripper_geom_id and contact.geom2 in hole_geom_id)
                        or (contact.geom2 == gripper_geom_id and contact.geom1 in hole_geom_id)):
                    return True
        return False

    def control_plotter(self):

        if self.overlap_time is None:
            self.overlap_time = 0

    def next_spiral(self, theta_current):
        # according to the article to assure successful insertion: p<=2d
        # where p is distance between consequent rings and d is clearance in centralized peg
        dt = 0.002
        v = 0.0015  # 0.0025/1.5
        p = 0.0006  # distance between the consecutive rings

        theta_dot_current = (2 * np.pi * v) / (p * np.sqrt(1 + theta_current ** 2))
        # todo: change +/- depending on the desired direction
        theta_next = theta_current + theta_dot_current * dt

        radius_next = (p / (2 * np.pi)) * theta_next

        x_next = radius_next * np.cos(theta_next)
        y_next = radius_next * np.sin(theta_next)

        return theta_next, radius_next, x_next, y_next
    def next_circle(self, theta_current):
        # overlap=error-2*radius_of_circle
        time_per_circle = 6
        dt = 0.002
        # 0.0025, 0.0026, 0.0027 with perturbation
        # 0.0028 always enters
        radius_of_circle = 0.00265
        # increase in angle per dt
        d_theta = (2*np.pi * dt)/time_per_circle
        theta_next = theta_current + d_theta

        x_next = radius_of_circle * np.cos(theta_next) - radius_of_circle
        y_next = radius_of_circle * np.sin(theta_next)

        return theta_next, radius_of_circle, x_next, y_next

    def zone_checker(self):
        """Used for collecting labels for supervised learning of the DNN model"""
        hole = deepcopy(self.sim.data.get_body_xpos("hole_hole"))
        hole_x = hole[0]
        hole_y = hole[1]
        peg = deepcopy(self.ee_pos)
        peg_x = peg[0]
        peg_y = peg[1]
        equation = np.sqrt((peg_x - hole_x) ** 2 + (peg_y - hole_y) ** 2)
        # print(equation)
        if equation < ERROR_TOP:  # peg_radius: #hole_radius:
            '''overlap big enough to perform impedance'''
            case = 1
            return case
        elif ERROR_TOP <= equation:
            '''not big enough overlap for impedance'''
            case = 0
            return case
        else:
            print('ERROR!')
            breakpoint()

    def circle_check(self):
        hole = deepcopy(self.sim.data.get_body_xpos("hole_hole"))
        hole_x = hole[0]
        hole_y = hole[1]
        peg = deepcopy(self.ee_pos)
        peg_x = peg[0]
        peg_y = peg[1]

        # print(np.sqrt((peg_x - hole_x) ** 2 + (peg_y - hole_y) ** 2))
        if np.sqrt((peg_x - hole_x) ** 2 + (peg_y - hole_y) ** 2) < 0.4/1000 and self.madeContact:
            self.stop = True

        if np.sqrt((peg_x-hole_x)**2+(peg_y-hole_y)**2) < ERROR_TOP:
            # print(np.sqrt((peg_x - hole_x) ** 2 + (peg_y - hole_y) ** 2))
            return True
        else:
            return False

