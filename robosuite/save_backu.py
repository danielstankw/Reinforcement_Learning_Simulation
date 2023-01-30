import pickle
import timeit

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
IMPEDANCE_MODES = {"fixed", "variable", "variable_kp"}
ERROR_TOP = 0.0007
PEG_RADIUS = 0.0021
HOLE_RADIUS = 0.0024
CIRCLE = True

class ImpedanceWithSpiral(Controller):
    """
    Controller for controlling robot arm via operational space control. Allows position and / or orientation control
    of the robot's end effector. For detailed information as to the mathematical foundation for this controller, please
    reference **add***

    NOTE: Control input actions can either be taken to be relative to the current position / orientation of the
    end effector or absolute values. In either case, a given action to this controller is assumed to be of the form:
    (x, y, z, ax, ay, az) if controlling pos and ori or simply (x, y, z) if only controlling pos

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)
        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:
            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities
        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range
        input_max (float or Iterable of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller
        input_min (float or Iterable of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or Iterable of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or Iterable of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller
        kp (float or Iterable of float): positional gain for determining desired torques based upon the pos / ori error.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)
        damping_ratio (float or Iterable of float): used in conjunction with kp to determine the velocity gain for
            determining desired torques based upon the joint pos errors. Can be either be a scalar (same value for all
            action dims), or a list (specific values for each dim)
        impedance_mode (str): Impedance mode with which to run this controller. Options are {"fixed", "variable",
            "variable_kp"}. If "fixed", the controller will have fixed kp and damping_ratio values as specified by the
            @kp and @damping_ratio arguments. If "variable", both kp and damping_ratio will now be part of the
            controller action space, resulting in a total action space of (6 or 3) + 6 * 2. If "variable_kp", only kp
            will become variable, with damping_ratio fixed at 1 (critically damped). The resulting action space will
            then be (6 or 3) + 6.
        kp_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is set to either
            "variable" or "variable_kp". This sets the corresponding min / max ranges of the controller action space
            for the varying kp values. Can be either be a 2-list (same min / max for all kp action dims), or a 2-list
            of list (specific min / max for each kp dim)
        damping_ratio_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is
            set to "variable". This sets the corresponding min / max ranges of the controller action space for the
            varying damping_ratio values. Can be either be a 2-list (same min / max for all damping_ratio action dims),
            or a 2-list of list (specific min / max for each damping_ratio dim)
        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller
        position_limits (2-list of float or 2-list of Iterable of floats): Limits (m) below and above which the
            magnitude of a calculated goal eef position will be clipped. Can be either be a 2-list (same min/max value
            for all cartesian dims), or a 2-list of list (specific min/max values for each dim)
        orientation_limits (2-list of float or 2-list of Iterable of floats): Limits (rad) below and above which the
            magnitude of a calculated goal eef orientation will be clipped. Can be either be a 2-list
            (same min/max value for all joint dims), or a 2-list of list (specific min/mx values for each dim)
        interpolator_pos (Interpolator): Interpolator object to be used for interpolating from the current position to
            the goal position during each timestep between inputted actions
        interpolator_ori (Interpolator): Interpolator object to be used for interpolating from the current orientation
            to the goal orientation during each timestep between inputted actions
        control_ori (bool): Whether inputted actions will control both pos and ori or exclusively pos
        uncouple_pos_ori (bool): Whether to decouple torques meant to control pos and torques meant to control ori
        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error
    Raises:
        AssertionError: [Invalid impedance mode]
    """

    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 input_max=1,
                 input_min=-1,
                 output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
                 output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
                 kp=150,
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
                 use_impedance=True,
                 use_spiral=True,
                 **kwargs  # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )

        self.overlap = False
        self.model = keras.models.load_model("keras_models/final_model_1")
        # self.model = pickle.load(open('RFClassifier.sav', 'rb'))
        self.threshold_precision = 0.866

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
        self.use_impedance = use_impedance

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

        # Control dimension
        # self.control_dim = 6 if self.use_ori else 3
        self.control_dim = control_dim
        # self.name_suffix = "POSE" if self.use_ori else "POSITION"

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

        # clip kp and kd
        kp = np.clip(kp, self.kp_min, self.kp_max)

        self.kp = self.nums2array(kp, 6)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio

        # -------- Elad PD params-------------
        # Kp_pos = 1 * 4500 * np.ones(3)  # 5*4500*np.ones(3)
        # Kp_ori = 1 * 100 * np.ones(3)  # 5*100*np.ones(3)
        # self.kp = np.concatenate((Kp_pos, Kp_ori))
        # Kd_pos = 1 * 5 * 0.707 * 2 * np.sqrt(Kp_pos)  # 5*0.707*2*np.sqrt(Kp_pos)
        # Kd_ori = 0.707 * 2 * np.sqrt(Kp_ori)  # 2*0.707*2*np.sqrt(Kp_ori)
        # self.kd = np.concatenate((Kd_pos, Kd_ori))

        self.kp_impedance = deepcopy(self.kp)
        self.kd_impedance = deepcopy(self.kd)

        # Verify the proposed impedance mode is supported
        assert impedance_mode in IMPEDANCE_MODES, "Error: Tried to instantiate IM_PB controller for unsupported " \
                                                  "impedance mode! Inputted impedance mode: {}, Supported modes: {}". \
            format(impedance_mode, IMPEDANCE_MODES)

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
            self.measured_sensor_bias = True

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
            print('%%%%%%%%%% Hole contact established %%%%%%%%%%%')

        # apply_ml = True
        #
        # if apply_ml:
        #     """
        #     Use ML model to predict prob of belonging to one of two classes:
        #         0: no overlap
        #         1: sufficient overlap for impedance insertion
        #     """
        #     # feature_names = ['Fz', 'Mx', 'My']
        #     if self.madeContact and not self.insertion:
        #         x = self.F_int[2:5].reshape(1, 3)
        #         y_predict = self.model(x)
        #         print('y_predict', y_predict)
        #         self.overlap = (y_predict >= self.threshold_precision)
        #         print('Model detected overlap', self.overlap)

        # if self.overlap and self.overlap_time is None and self.existsOverlap is False and self.madeContact:
        #     self.overlap_time = self.sim.data.time  # collects time of first overlap
        #     self.existsOverlap = True  # tells us if overlap was detected and keeps that
        #     # mode on, so we do not enter this code again
        #     # TODO: stop at overlap: SPIRAL_FLAG = False
        #     self.spiral_flag = False

        case = self.zone_checker()
        # if case == 1:
        #     print('Impedance ')
        # elif case == 0:
        #     print('Not big enough overlap')
        # print('Case', case)
        self.zones.append(case)
        # if sum(self.zones) > 4:
        #     breakpoint()

        if self.circle_check() and self.overlap_time is None and self.existsOverlap is False and self.madeContact:
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
            self.existsOverlap = True
            # TODO: if stop at the overlap: SPIRAL_FLAG=False
            self.spiral_flag = False

        # we don't want to do spiral search in the hole, so if peg tip
        # goes below surface level of hole than we turn off spiral search
        if self.madeContact:
            if self.spiral_flag:
                """If contact was made and we use spiral search -> update the trajectory"""
                if CIRCLE:
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
                print('Spiral')

            # if conditions for spiral are not satisfied but the contact has been made
            else:
                print("No Spiral")

            if self.measured_sensor_bias is True:
                print('Initializing contact PD params')
                self.impedance_vec = deepcopy(self.desired_pos)
                # we set self.measured_sensor_bias = None, so that we enter that code only ONCE!
                self.measured_sensor_bias = None
                # self.kp = deepcopy(np.clip(self.kp_impedance, self.kp_min, self.kp_max))
                # self.kd = deepcopy(np.clip(self.kd_impedance, 0.0, 4 * 2 * np.sqrt(self.kp) * np.sqrt(2)))
                # TODO the initial PD parameters weren't great for the spiral so we use 3 sets of params!
                self.kp = np.array([15000.0, 15000.0, 250.0, 450.0, 450.0, 450.0])
                self.kd = 2 * np.sqrt(self.kp) * np.sqrt(2)

            if self.spiral_flag is False:
                '''If spiral search is/becomes disabled'''
                # TODO Impedance/PD at contact uses 3rd set of PD parameters
                self.kp = np.array([5000.0, 5000.0, 250.0, 450.0, 450.0, 450.0])
                self.kd = 2 * np.sqrt(self.kp) * np.sqrt(2)
                # self.kp = deepcopy(np.clip(self.kp_impedance, self.kp_min, self.kp_max))
                # self.kd = deepcopy(np.clip(self.kd_impedance, 0.0, 4 * 2 * np.sqrt(self.kp) * np.sqrt(2)))
                """
                wait_time: wait time once overlap was detected, to stabilize sensor reading
                """
                wait_time = 3.0
                self.end_wait = self.overlap_time + wait_time
                self.insertion = True  # Added to stop making predictions once we reach overlap stage

                if self.sim.data.time - self.overlap_time >= wait_time:
                    if self.use_impedance:
                        '''After waiting for wait_time seconds we use impedance for final stage of insertion'''
                        print('Using Impedance')
                        self.desired_pos = deepcopy(
                            self.ImpedanceEq(self.F_int, self.F0, self.desired_pos[:3], self.desired_pos[3:6],
                                             self.desired_pos[6:9], self.desired_pos[9:12],
                                             self.sim.model.opt.timestep))
                    # else use PD
                    else:
                        print('Using PD')
                else:
                    '''After overlap between peg and a hole happen, we wait for WAIT_TIME seconds'''
                    print(f'Pausing for more {round(wait_time-(self.sim.data.time - self.overlap_time), 4)} out of {wait_time} sec')
        # TODO: for circle comment
        self.desired_pos[:2] += np.array([self.x_spiral_next, self.y_spiral_next])

        ori_real = T.Rotation_Matrix_To_Vector(self.final_orientation, self.ee_ori_mat)

        print('------------------------------')
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

        decoupled_wrench = np.concatenate([desired_force, desired_torque])
        self.torques = np.dot(self.J_full.T, decoupled_wrench).reshape(6, ) + self.torque_compensation

        self.set_desired_goal = False
        # Always run superclass call for any cleanups at the end
        super().run_controller()
        if np.isnan(self.torques).any():
            self.torques = np.zeros(6)

        self.plotter = True
        if self.plotter:

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

    def save_plot_data(self):
        data = {}
        data["time"] = self.time

        data["Xm position"] = self.impedance_model_pos_vec_x
        data["Xr position"] = self.ee_pos_vec_x
        data["X_ref position"] = self.pos_min_jerk_x

        data["Ym position"] = self.impedance_model_pos_vec_y
        data["Yr position"] = self.ee_pos_vec_y
        data["Y_ref position"] = self.pos_min_jerk_y

        data["Zm position"] = self.impedance_model_pos_vec_z
        data["Zr position"] = self.ee_pos_vec_z
        data["Z_ref position"] = self.pos_min_jerk_z

        data["Fx"] = self.wernce_vec_int_Fx
        data["Fx_des"] = self.desired_force_x

        data["Fy"] = self.wernce_vec_int_Fy
        data["Fy_des"] = self.desired_force_y

        data["Fz"] = self.wernce_vec_int_Fz
        data["Fz_des"] = self.desired_force_z

        data["Mx"] = self.wernce_vec_int_Mx
        data["mx_des"] = self.desired_torque_x

        data["My"] = self.wernce_vec_int_My
        data["my_des"] = self.desired_torque_y

        data["Mz"] = self.wernce_vec_int_Mz
        data["mz_des"] = self.desired_torque_z

        df = pd.DataFrame(data)
        # df.to_csv("data_daniel_plus_y_1mm.csv", index=False)

    def control_plotter(self):
        # name='counterClockwise_no stopping_-7'
        # self.save_plot_data()
        t = self.time_vec  # list(range(0, np.size(self.ee_pos_vec_x)))
        # ################################################################################################################
        # idx = int(6 / 0.002)
        # idx2 = int(7.3 / 0.002)
        # print('min_jerk start', self.pos_min_jerk_x[0] * 1000)
        # print('min_jerk end', self.pos_min_jerk_x[-1] * 1000)
        # print('time', t[idx])
        # print('minimum_jerk_x', self.pos_min_jerk_x[idx] * 1000)
        # print('robot_x', self.ee_pos_vec_x[idx] * 1000)
        # print('diff', abs(self.pos_min_jerk_x[idx] - self.ee_pos_vec_x[idx]) * 1000)
        #
        # print('time', t[idx2])
        # print('minimum_jerk_x', self.pos_min_jerk_x[idx2] * 1000)
        # print('robot_x', self.ee_pos_vec_x[idx2] * 1000)
        # print('diff', abs(self.pos_min_jerk_x[idx2] - self.ee_pos_vec_x[idx2]) * 1000)
        if self.overlap_time is None:
            self.overlap_time = 0
        plt.figure("Spiral")
        plt.plot(self.spiral_x, self.spiral_y, 'g', label='Ref position')
        plt.plot(self.robot_spiral_x, self.robot_spiral_y, 'b', label='Robot position')
        hole = self.sim.data.get_body_xpos("hole_hole")
        plt.plot(hole[0], hole[1], "ro")
        plt.plot(self.spiral_x[0], self.spiral_y[0], "go")
        plt.plot(self.robot_spiral_x[0], self.robot_spiral_y[0], "bo")
        plt.legend()
        plt.grid()
        # plt.savefig(f"/home/user/Desktop/Simulation_measurements/{name}/Spiral.png")

        plt.figure()
        ax1 = plt.subplot(311)
        ax1.plot(self.pos_min_jerk_x, self.pos_min_jerk_y, 'g', label='Ref position')
        ax1.plot(self.ee_pos_x_vec, self.ee_pos_y_vec, 'b', label='Robot position')
        ax1.set_ylabel('X')
        ax1.set_xlabel('Y')

        ax2 = plt.subplot(312)
        ax2.plot(t, self.pos_min_jerk_x, 'g--', label='X_ref position')
        ax2.plot(t, self.ee_pos_x_vec, 'b', label='Xr position')
        ax2.legend()
        ax2.set_title('X Position [m]')

        ax3 = plt.subplot(313)
        ax3.plot(t, self.pos_min_jerk_y, 'g--', label='Y_ref position')
        ax3.plot(t, self.ee_pos_y_vec, 'b', label='Yr position')
        ax3.legend()
        ax3.set_title('Y Position [m]')

        plt.figure("Position")
        ax1 = plt.subplot(311)
        ax1.plot(t, self.impedance_pos_vec_x, 'g--', label='Xm position')
        ax1.plot(t, self.ee_pos_x_vec, 'b', label='Xr position')
        ax1.plot(t, self.pos_min_jerk_x, 'r--', label='X_ref position')
        ax1.axvline(x=self.overlap_time, color='k')
        ax1.axvline(x=self.end_wait, color='k')
        ax1.axvline(x=self.initialContactTime, color='r')
        ax1.legend()
        ax1.set_title('X Position [m]')

        ax2 = plt.subplot(312)
        ax2.plot(t, self.impedance_pos_vec_y, 'g--', label='Ym position')
        ax2.plot(t, self.ee_pos_y_vec, 'b', label='Yr position')
        ax2.plot(t, self.pos_min_jerk_y, 'r--', label='Y_ref position')
        ax2.axvline(x=self.overlap_time, color='k')
        ax2.axvline(x=self.end_wait, color='k')
        ax2.axvline(x=self.initialContactTime, color='r')
        ax2.legend()
        ax2.set_title('Y Position [m]')

        ax3 = plt.subplot(313)
        ax3.plot(t, self.impedance_pos_vec_z, 'g--', label='Zm position')
        ax3.plot(t, self.ee_pos_z_vec, 'b', label='Zr position')
        ax3.plot(t, self.pos_min_jerk_z, 'r--', label='Z_ref position')
        ax3.axvline(x=self.overlap_time, color='k')
        ax3.axvline(x=self.end_wait, color='k')
        ax3.axvline(x=self.initialContactTime, color='r')
        ax3.legend()
        ax3.set_title('Z Position [m]')
        # plt.savefig(f"/home/user/Desktop/Simulation_measurements/{name}/Position.png")
        ################################################################################################################
        plt.figure("Linear velocity")
        ax1 = plt.subplot(311)
        ax1.plot(t, self.impedance_vel_vec_x, 'g--', label='Xm vel')
        ax1.plot(t, self.ee_vel_x_vec, 'b', label='Xr vel')
        ax1.plot(t, self.vel_min_jerk_x, 'r--', label='X_ref vel')
        ax1.axvline(x=self.overlap_time, color='k')
        ax1.axvline(x=self.end_wait, color='k')
        ax1.axvline(x=self.initialContactTime, color='r')
        ax1.legend()
        ax1.set_title('X Velocity [m/s]')

        ax2 = plt.subplot(312)
        ax2.plot(t, self.impedance_vel_vec_y, 'g--', label='Ym vel')
        ax2.plot(t, self.ee_vel_y_vec, 'b', label='Yr vel')
        ax2.plot(t, self.vel_min_jerk_y, 'r--', label='Y_ref vel')
        ax2.axvline(x=self.overlap_time, color='k')
        ax2.axvline(x=self.end_wait, color='k')
        ax2.axvline(x=self.initialContactTime, color='r')
        ax2.legend()
        ax2.set_title('Y Velocity [m/s]')

        ax3 = plt.subplot(313)
        ax3.plot(t, self.impedance_vel_vec_z, 'g--', label='Zm vel')
        ax3.plot(t, self.ee_vel_z_vec, 'b', label='Zr vel')
        ax3.plot(t, self.vel_min_jerk_z, 'r--', label='Z_ref vel')
        ax3.axvline(x=self.overlap_time, color='k')
        ax3.axvline(x=self.end_wait, color='k')
        ax3.axvline(x=self.initialContactTime, color='r')
        ax3.legend()
        ax3.set_title('Z Velocity [m/s]')
        # plt.savefig(f"/home/user/Desktop/Simulation_measurements/{name}/LinearVelocity.png")
        ################################################################################################################
        plt.figure("Angular Velocity")
        ax1 = plt.subplot(311)
        ax1.plot(t, self.ee_ori_vel_x_vec, 'b', label='Xr')
        ax1.plot(t, self.ori_vel_min_jerk_x, 'r--', label='X_ref ')
        ax1.legend()
        ax1.set_title('X ori vel [rad/s]')

        ax2 = plt.subplot(312)
        ax2.plot(t, self.ee_ori_vel_y_vec, 'b', label='Yr ')
        ax2.plot(t, self.ori_vel_min_jerk_y, 'r--', label='Y_ref ')
        ax2.legend()
        ax2.set_title('Y ori vel [rad/s]')

        ax3 = plt.subplot(313)
        ax3.plot(t, self.ee_ori_vel_z_vec, 'b', label='Zr ')
        ax3.plot(t, self.ori_vel_min_jerk_z, 'r--', label='Z_ref ')
        ax3.legend()
        ax3.set_title('Z ori vel [rad/s]')
        ################################################################################################################
        plt.figure("Orientation")
        ax1 = plt.subplot(311)
        ax1.plot(t, self.ee_ori_x_vec, 'b', label='Xr')
        ax1.plot(t, self.ori_min_jerk_x, 'r', label='X_ref ')
        ax1.plot(t, self.impedance_ori_vec_x, 'g--', label='Xm ')
        ax1.legend()
        ax1.set_title('X ori [rad]')

        ax2 = plt.subplot(312)
        ax2.plot(t, self.ee_ori_y_vec, 'b', label='Yr ')
        ax2.plot(t, self.ori_min_jerk_y, 'r', label='Y_ref ')
        ax2.plot(t, self.impedance_ori_vec_y, 'g--', label='Ym ')
        ax2.legend()
        ax2.set_title('Y ori [rad]')

        ax3 = plt.subplot(313)
        ax3.plot(t, self.ee_ori_z_vec, 'b', label='Zr ')
        ax3.plot(t, self.ori_min_jerk_z, 'r', label='Z_ref ')
        ax3.plot(t, self.impedance_ori_vec_z, 'g--', label='Zm ')
        ax3.legend()
        ax3.set_title('Z ori[rad]')
        ################################################################################################################
        plt.figure("Forces")
        ax1 = plt.subplot(311)
        ax1.plot(t, self.sensor_fx, 'b', label='Fx_sensor')
        ax1.plot(t, self.applied_wrench_fx, 'g', label='Fx_wrench')
        ax1.axvline(x=self.overlap_time, color='k')
        ax1.axvline(x=self.end_wait, color='k')
        ax1.axvline(x=self.initialContactTime, color='r')
        ax1.legend()
        ax1.set_title('Fx [N]')

        ax2 = plt.subplot(312)
        ax2.plot(t, self.sensor_fy, 'b', label='Fy_sensor')
        ax2.plot(t, self.applied_wrench_fy, 'g', label='Fy_wrench')
        ax2.axvline(x=self.overlap_time, color='k')
        ax2.axvline(x=self.end_wait, color='k')
        ax2.axvline(x=self.initialContactTime, color='r')
        ax2.legend()
        ax2.set_title('Fy [N]')

        ax3 = plt.subplot(313)
        ax3.plot(t, self.sensor_fz, 'b', label='Fz_sensor')
        ax3.plot(t, self.applied_wrench_fz, 'g', label='Fz_wrench')
        ax3.axvline(x=self.overlap_time, color='k')
        ax3.axvline(x=self.end_wait, color='k')
        ax3.axvline(x=self.initialContactTime, color='r')
        ax3.legend()
        ax3.set_title('Fz [N]')
        # plt.savefig(f"/home/user/Desktop/Simulation_measurements/{name}/Forces.png")
        ################################################################################################################
        plt.figure("Moments")
        ax1 = plt.subplot(311)
        ax1.plot(t, self.sensor_mx, 'b', label='Mx_sensor')
        ax1.plot(t, self.applied_wrench_mx, 'g', label='Mx_wrench')
        ax1.axvline(x=self.overlap_time, color='k')
        ax1.axvline(x=self.end_wait, color='k')
        ax1.axvline(x=self.initialContactTime, color='r')
        ax1.legend()
        ax1.set_title('Mx [Nm]')

        ax2 = plt.subplot(312)
        ax2.plot(t, self.sensor_my, 'b', label='My_sensor')
        ax2.plot(t, self.applied_wrench_my, 'g', label='My_wrench')
        ax2.axvline(x=self.overlap_time, color='k')
        ax2.axvline(x=self.end_wait, color='k')
        ax2.axvline(x=self.initialContactTime, color='r')
        ax2.legend()
        ax2.set_title('My [Nm]')

        ax3 = plt.subplot(313)
        ax3.plot(t, self.sensor_mz, 'b', label='Mz_sensor')
        ax3.plot(t, self.applied_wrench_mz, 'g', label='Mz_wrench')
        ax3.axvline(x=self.overlap_time, color='k')
        ax3.axvline(x=self.end_wait, color='k')
        ax3.axvline(x=self.initialContactTime, color='r')
        ax3.legend()
        ax3.set_title('Mz [Nm]')
        # plt.savefig(f"/home/user/Desktop/Simulation_measurements/{name}/Moments.png")

        plt.figure("Zones")
        plt.scatter(t, self.zones)
        plt.axvline(x=self.initialContactTime, color='r')
        plt.xlabel('Time [sec]')
        # plt.savefig(f"/home/user/Desktop/Simulation_measurements/{name}/Zones.png")
        #
        data = {}
        data["time"] = t
        data["Fx"] = self.sensor_fx
        data["Fy"] = self.sensor_fy
        data["Fz"] = self.sensor_fz
        data['Vx'] = self.ee_vel_x_vec
        data['Vy'] = self.ee_vel_y_vec
        data["Mx"] = self.sensor_mx
        data["My"] = self.sensor_my
        data["Case"] = self.zones
        data['t_contact'] = self.initialContactTime

        df = pd.DataFrame(data)
        # df.to_csv("circle_0.0007.csv", index=False)

        plt.show()

    def next_spiral(self, theta_current):
        # according to the article to assure successful insertion: p<=2d
        # where p is distance between consequent rings and d is clearance in centralized peg
        #v = 0.0009230 / 2   # total velocity (linear and angular)
        v = 0.0025/1.5
        p = 2 * 0.0003  # distance between the consecutive rings
        dt = 0.002

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
        radius_of_circle = 0.0026
        # increase in angle per dt
        d_theta = (2*np.pi * dt)/time_per_circle
        theta_next = theta_current + d_theta

        x_next = radius_of_circle * np.cos(theta_next) - radius_of_circle
        y_next = radius_of_circle * np.sin(theta_next)

        return theta_next, radius_of_circle, x_next, y_next

    def zone_checker(self):
        hole = deepcopy(self.sim.data.get_body_xpos("hole_hole"))
        hole_x = hole[0]
        hole_y = hole[1]
        peg = deepcopy(self.ee_pos)
        peg_x = peg[0]
        peg_y = peg[1]
        equation = np.sqrt((peg_x - hole_x) ** 2 + (peg_y - hole_y) ** 2)
        print(equation)
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
        if np.sqrt((peg_x-hole_x)**2+(peg_y-hole_y)**2) < ERROR_TOP:
            # print(np.sqrt((peg_x - hole_x) ** 2 + (peg_y - hole_y) ** 2))
            return True
        else:
            return False
