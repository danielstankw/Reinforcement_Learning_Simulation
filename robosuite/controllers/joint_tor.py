from robosuite.controllers.base_controller import Controller
import numpy as np
from scipy.spatial.transform import Rotation as R
from robosuite.utils.min_jerk_planner import PathPlan
import robosuite.utils.angle_transformation as at
import matplotlib.pyplot as plt
from copy import deepcopy
import robosuite.utils.transform_utils as T
from robosuite.utils.control_utils import *

class JointTorqueController(Controller):
    """
    Controller for controlling the robot arm's joint torques. As the actuators at the mujoco sim level are already
    torque actuators, this "controller" usually simply "passes through" desired torques, though it also includes the
    typical input / output scaling and clipping, as well as interpolator features seen in other controllers classes
    as well

    NOTE: Control input actions assumed to be taken as absolute joint torques. A given action to this
    controller is assumed to be of the form: (torq_j0, torq_j1, ... , torq_jn-1) for an n-joint robot
    """

    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 input_max=1,
                 input_min=-1,
                 output_max=0.05,
                 output_min=-0.05,
                 policy_freq=20,    # frequency at which actions are fed into this controller
                 torque_limits=None,
                 total_time=25,
                 trajectory_time_1=10,
                 trajectory_time_2=10,
                 plotter=True,
                 interpolator=None,
                 **kwargs  # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )

        # Controller PD gains
        Kp_pos = 1500 * np.ones(3)
        Kp_ori = 100 * np.ones(3)
        self.Kp = np.append(Kp_pos, Kp_ori)
        Kd_pos = 0.707 * 2 * np.sqrt(Kp_pos)
        Kd_ori = 0.707 * 2 * np.sqrt(Kp_ori)
        self.Kd = np.append(Kd_pos, Kd_ori)

        # Control dimension
        self.control_dim = len(joint_indexes["joints"])

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # limits (if not specified, set them to actuator limits by default)
        self.torque_limits = np.array(torque_limits) if torque_limits is not None else self.actuator_limits

        # control frequency
        self.control_freq = policy_freq
        self.dt = 1/self.control_freq
        self.gripper_val = np.array([0])

        # initialize torques
        self.goal_torque = None  # Goal torque desired, pre-compensation
        self.current_torque = np.zeros(self.control_dim)  # Current torques being outputted, pre-compensation
        self.torques = None  # Torques returned every time run_controller is called

        # initialize to 0
        self.pos_ref = np.zeros(3)
        self.Vrot_ref = np.zeros(3)
        self.lin_vel_ref = np.zeros(3)
        self.ang_vel_ref = np.zeros(3)
        self.wrench_command = np.zeros(6)
        # --- min_jerk-----

        self.t = 0
        self.time_const = 0
        self.total_sim_time = total_time
        self.time_for_trajectory_1 = trajectory_time_1
        self.time_for_trajectory_2 = trajectory_time_2
        self.total_traj_time =trajectory_time_1 + trajectory_time_2

        # for the graphs
        self.plotter = plotter
        self.time_plot = []
        self.ee_pos_x, self.ee_pos_y, self.ee_pos_z = [], [], []
        self.ee_vel_x, self.ee_vel_y, self.ee_vel_z = [], [], []
        self.ee_ori_x, self.ee_ori_y, self.ee_ori_z = [], [], []
        self.min_jerk_x, self.min_jerk_y, self.min_jerk_z = [], [], []
        self.min_jerk_vx, self.min_jerk_vy, self.min_jerk_vz = [], [], []
        self.min_jerk_ori_x, self.min_jerk_ori_y, self.min_jerk_ori_z = [], [], []
        self.wrench_fx, self.wrench_fy, self.wrench_fz = [], [], []
        self.wrench_mx, self.wrench_my, self.wrench_mz = [], [], []
        self.torque_1, self.torque_2, self.torque_3 = [], [], []
        self.torque_4, self.torque_5, self.torque_6 = [], [], []
        self.sensor_fx, self.sensor_fy, self.sensor_fz = [], [], []
        self.sensor_mx, self.sensor_my, self.sensor_mz = [], [], []

    def set_goal(self, action, set_goal_point = None, set_initial_point= None):
        """
        Sets goal for the run_controller()
        Args:
            action: parameter values
            t_inst: current time
        """
        # Update state
        self.update()
        """Based on base_controller.py obtain access to: 
        {ee_pos,ee_ori_mat,ee_pos_vel,ee_ori_vel,joint_pos,joint_vel,J_pos,J_ori,J_full,mass_matrix}"""

        # self.joint_index = [0, 1, 2, 3, 4, 5]
        # self.qpos_index = [0, 1, 2, 3, 4, 5]
        # self.qvel_index = [0, 1, 2, 3, 4, 5]

        # --------- Compute desired force and torque based on errors --------
        self.planner = PathPlan(set_initial_point, set_goal_point, self.total_traj_time)
        self.planner.built_min_jerk_traj()
        self.t_bias = self.t
        if self.t_bias < 0:
            self.t_bias = 0


    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        self.t = self.sim.data.time
        t_now = self.t - self.t_bias

        [pos_ref, ori_ref, vel_ref, ori_vel_ref] = self.planner.trajectory_planning(t_now)
        self.trajectory_values.append([pos_ref, ori_ref])
        # Get real robot measurements
        ori_error = orientation_error(T.euler2mat(ori_ref), self.ee_ori_mat)

        # Define the desired pose and calculate the tracking errors
        pos_error = pos_ref - self.ee_pos
        lin_vel_pos_error = vel_ref - self.ee_pos_vel
        ang_vel_error = ori_vel_ref - self.ee_ori_vel

        # Calculate wrench commands using PD control law
        desired_force = self.Kp[:3] * pos_error + self.Kd[:3] * lin_vel_pos_error
        desired_torque = self.Kp[3:] * ori_error + self.Kd[3:] * ang_vel_error
        self.wrench_command = np.append(desired_force, desired_torque)

        self.goal_torque = np.dot(self.J_full.T, self.wrench_command)

        # Update state
        self.update()

        # Add gravity compensation
        self.torques = np.array(self.goal_torque) + self.torque_compensation

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        # plotting section
        if self.plotter:
            if self.t >= (self.total_sim_time - (self.dt + 0.01)):
                print(f'PLOTTING AT {self.t}')
                self.control_plotter()

            self.time_plot.append(self.t)
            self.ee_pos_x.append(self.ee_pos[0])
            self.ee_pos_y.append(self.ee_pos[1])
            self.ee_pos_z.append(self.ee_pos[2])
            self.ee_vel_x.append(self.ee_pos_vel[0])
            self.ee_vel_y.append(self.ee_pos_vel[1])
            self.ee_vel_z.append(self.ee_pos_vel[2])
            rotvec = R.from_matrix(self.ee_ori_mat).as_rotvec()
            self.ee_ori_x.append(rotvec[0])
            self.ee_ori_y.append(rotvec[1])
            self.ee_ori_z.append(rotvec[2])
            self.min_jerk_x.append(pos_ref[0])
            self.min_jerk_y.append(pos_ref[1])
            self.min_jerk_z.append(pos_ref[2])
            self.min_jerk_vx.append(self.lin_vel_ref[0])
            self.min_jerk_vy.append(self.lin_vel_ref[1])
            self.min_jerk_vz.append(self.lin_vel_ref[2])
            self.min_jerk_ori_x.append(self.Vrot_ref[0])
            self.min_jerk_ori_y.append(self.Vrot_ref[1])
            self.min_jerk_ori_z.append(self.Vrot_ref[2])
            self.wrench_fx.append(self.wrench_command[0])
            self.wrench_fy.append(self.wrench_command[1])
            self.wrench_fz.append(self.wrench_command[2])
            self.wrench_mx.append(self.wrench_command[3])
            self.wrench_my.append(self.wrench_command[4])
            self.wrench_mz.append(self.wrench_command[5])
            self.torque_1.append(self.torques[0])
            self.torque_2.append(self.torques[1])
            self.torque_3.append(self.torques[2])
            self.torque_4.append(self.torques[3])
            self.torque_5.append(self.torques[4])
            self.torque_6.append(self.torques[5])
            # self.sensor_fx.append(self.sensor_reading[0])
            # self.sensor_fy.append(self.sensor_reading[1])
            # self.sensor_fz.append(self.sensor_reading[2])
            # self.sensor_mx.append(self.sensor_reading[3])
            # self.sensor_my.append(self.sensor_reading[4])
            # self.sensor_mz.append(self.sensor_reading[5])

        # Return final torques
        return self.torques

    def reset_goal(self):
        """
        Resets joint torque goal to be all zeros (pre-compensation)
        """
        self.goal_torque = np.zeros(self.control_dim)

    @property
    def name(self):
        return 'JOINT_TORQUE'

    def control_plotter(self):
        # ---------------- position ------------------

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        fig.suptitle('Position Plots')
        ax1.plot(self.time_plot, self.ee_pos_x, label='Xr position')
        ax1.plot(self.time_plot, self.min_jerk_x, label='X_ref position')
        ax1.legend()
        ax1.grid()
        ax2.plot(self.time_plot, self.ee_pos_y, label='Yr position')
        ax2.plot(self.time_plot, self.min_jerk_y, label='Y_ref position')
        ax2.legend()
        ax2.grid()
        ax3.plot(self.time_plot, self.ee_pos_z, label='Zr position')
        ax3.plot(self.time_plot, self.min_jerk_z, label='Z_ref position')
        ax3.legend()
        ax3.grid()
        fig.text(0.5, 0.04, 'Time [sec]', ha='center')
        fig.text(0.04, 0.5, 'Position [m]', va='center', rotation='vertical')

        # ---------------- linear velocity ------------------
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        fig.suptitle('Linear Velocity Plots')
        ax1.plot(self.time_plot, self.ee_vel_x, label='Vx velocity')
        ax1.plot(self.time_plot, self.min_jerk_vx, label='Vx_ref velocity')
        ax1.legend()
        ax1.grid()
        ax2.plot(self.time_plot, self.ee_vel_y, label='Vy velocity')
        ax2.plot(self.time_plot, self.min_jerk_vy, label='Vy_ref velocity')
        ax2.legend()
        ax2.grid()
        ax3.plot(self.time_plot, self.ee_vel_z, label='Vz velocity')
        ax3.plot(self.time_plot, self.min_jerk_vz, label='Vz_ref velocity')
        ax3.legend()
        ax3.grid()
        fig.text(0.5, 0.04, 'Time [sec]', ha='center')
        fig.text(0.04, 0.5, 'Velocity [m/s]', va='center', rotation='vertical')


        # ---------------- rotation ------------------
        # plt.figure()
        # plt.plot(self.time_plot, self.ee_ori_x, label='Rx rotation')
        # plt.plot(self.time_plot, self.min_jerk_ori_x, label='Rx_ref rotation')
        # plt.legend()
        # plt.ylabel('Rotation [rad]')
        #
        # plt.figure()
        # plt.plot(self.time_plot, self.ee_ori_y, label='Ry rotation')
        # plt.plot(self.time_plot, self.min_jerk_ori_y, label='Ry_ref rotation')
        # plt.legend()
        # plt.ylabel('Rotation [rad]')
        #
        # plt.figure()
        # plt.plot(self.time_plot, self.ee_ori_z, label='Rz rotation')
        # plt.plot(self.time_plot, self.min_jerk_ori_z, label='Rz_ref rotation')
        # plt.legend()
        # plt.ylabel('Rotation [rad]')
        # ----------------- wrench ----------------------
        # plt.figure()
        # plt.plot(self.time_plot, self.wrench_fx, label='Fx')
        # plt.legend()
        # plt.ylabel('Wrench Force [N]')
        #
        # plt.figure()
        # plt.plot(self.time_plot, self.wrench_fy, label='Fy')
        # plt.legend()
        # plt.ylabel('Wrench Force [N]')
        #
        # plt.figure()
        # plt.plot(self.time_plot, self.wrench_fy, label='Fz')
        # plt.legend()
        # plt.ylabel('Wrench force [N]')
        #
        # plt.figure()
        # plt.plot(self.time_plot, self.wrench_mx, label='Mx')
        # plt.legend()
        # plt.ylabel('Wrench torque [Nm]')
        #
        # plt.figure()
        # plt.plot(self.time_plot, self.wrench_my, label='My')
        # plt.legend()
        # plt.ylabel('Wrench torque [Nm]')
        #
        # plt.figure()
        # plt.plot(self.time_plot, self.wrench_mz, label='Mz')
        # plt.legend()
        # plt.ylabel('Wrench torque [Nm]')
        # plt.show()
        # ------------sensor --------------
        # plt.figure()
        # plt.plot(self.time_plot, self.sensor_fx, label='Fx')
        # plt.legend()
        # plt.grid()
        # plt.ylabel('Sensor Fx [N]')
        #
        # plt.figure()
        # plt.plot(self.time_plot, self.sensor_fy, label='Fy')
        # plt.legend()
        # plt.grid()
        # plt.ylabel('Sensor Fy [N]')
        #
        # plt.figure()
        # plt.plot(self.time_plot, self.sensor_fz, label='Fz')
        # plt.legend()
        # plt.grid()
        # plt.ylabel('Sensor Fz [N]')
        #
        # plt.figure()
        # plt.plot(self.time_plot, self.sensor_mx, label='mx')
        # plt.legend()
        # plt.grid()
        # plt.ylabel('Sensor Mx [N]')
        #
        # plt.figure()
        # plt.plot(self.time_plot, self.sensor_my, label='my')
        # plt.legend()
        # plt.grid()
        # plt.ylabel('Sensor My [N]')
        #
        # plt.figure()
        # plt.plot(self.time_plot, self.sensor_mz, label='mz')
        # plt.legend()
        # plt.grid()
        # plt.ylabel('Sensor Mz [N]')
        plt.show()