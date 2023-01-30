"""
Minimum jerk trajectory for 6DOF robot
Latest update: 14.12.2020
Written by Daniel Stankowski
"""
import numpy as np
import robosuite.utils.angle_transformation as at

class PathPlan(object):
    """
    IMPORTANT: When the pose is passed [x,y,z,Rx,Ry,Rz] one has to convert the orientation
    part from axis-angle representation to the Euler before running this script.
    """

    def __init__(self, initial_pos, initial_ori, target_pos, target_ori, total_time):
        self.initial_position = initial_pos
        self.target_position = target_pos
        self.initial_orientation = initial_ori
        self.target_orientation = target_ori
        self.tfinal = total_time

    def trajectory_planning(self, t):
        X_init = self.initial_position[0]
        Y_init = self.initial_position[1]
        Z_init = self.initial_position[2]

        X_final = self.target_position[0]
        Y_final = self.target_position[1]
        Z_final = self.target_position[2]

        # position

        x_traj = (X_final - X_init) / (self.tfinal ** 3) * (6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + X_init
        y_traj = (Y_final - Y_init) / (self.tfinal ** 3) * (6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + Y_init
        z_traj = (Z_final - Z_init) / (self.tfinal ** 3) * (6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + Z_init
        position = np.array([x_traj, y_traj, z_traj])

        # velocities
        vx = (X_final - X_init) / (self.tfinal ** 3) * (30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
        vy = (Y_final - Y_init) / (self.tfinal ** 3) * (30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
        vz = (Z_final - Z_init) / (self.tfinal ** 3) * (30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
        velocity = np.array([vx, vy, vz])

        # acceleration
        ax = (X_final - X_init) / (self.tfinal ** 3) * (120 * (t ** 3) / (self.tfinal ** 2) - 180 * (t ** 2) / self.tfinal + 60 * t)
        ay = (Y_final - Y_init) / (self.tfinal ** 3) * (120 * (t ** 3) / (self.tfinal ** 2) - 180 * (t ** 2) / self.tfinal + 60 * t)
        az = (Z_final - Z_init) / (self.tfinal ** 3) * (120 * (t ** 3) / (self.tfinal ** 2) - 180 * (t ** 2) / self.tfinal + 60 * t)
        acceleration = np.array([ax, ay, az])

        #   -----------------------------------rotation (based on rotation matrices) ---------------------------------------
        vec_x = self.initial_orientation[0]
        vec_y = self.initial_orientation[1]
        vec_z = self.initial_orientation[2]

        # alpha_final = self.target_orientation[0]
        # beta_final = self.target_orientation[1]
        # gamma_final = self.target_orientation[2]

        Vrot = np.array([vec_x, vec_y, vec_z])
        magnitude, direction = at.Axis2Vector(Vrot)
        # In case of lack of rotation:
        lower_bound = 10e-6
        if magnitude < lower_bound:
            magnitude = 0.0
            direction = np.array([0.0, 0.0, 0.0])

        magnitude_traj = (0 - magnitude) / (self.tfinal ** 3) * (6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + magnitude
        #   we want to decrease the magnitude of the rotation from some initial value to 0
        vec_x_traj = magnitude_traj*direction[0]
        vec_y_traj = magnitude_traj*direction[1]
        vec_z_traj = magnitude_traj*direction[2]

        orientation = np.array([vec_x_traj, vec_y_traj, vec_z_traj])

        # angular velocities
        # alpha_d_traj = (alpha_final - vec_x) / (self.tfinal ** 3) * (30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
        # beta_d_traj = (beta_final - beta_init) / (self.tfinal ** 3) * (30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
        # gamma_d_traj = (gamma_final - gamma_init) / (self.tfinal ** 3) * (30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))

        magnitude_vel_traj = (0 - magnitude) / (self.tfinal ** 3) * (30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
        vec_x_d_traj = magnitude_vel_traj * direction[0]
        vec_y_d_traj = magnitude_vel_traj * direction[1]
        vec_z_d_traj = magnitude_vel_traj * direction[2]

        ang_vel = np.array([vec_x_d_traj, vec_y_d_traj, vec_z_d_traj])

        return [position, orientation, velocity, ang_vel, acceleration]
