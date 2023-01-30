"""
UR robot uses an axis-angle convention to describe the orientation Rx,Ry,Rz i.e rotation vector
To implement a minimum-jerk trajectory we need to convert the angle to Euler angles
Notion used is "RPY" roll-pitch-yaw convention i.e. XYZ Euler representation
http://web.mit.edu/2.05/www/Handout/HO2.PDF
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
Last modified: 28.12.2020
Daniel Stankowski
Elad Newman
"""

from scipy.spatial.transform import Rotation as R
import numpy as np


def Robot2Euler(orientation):
    """
    Convert axis-angle to euler X-Y-Z convention
    :param orientation: np.array([Rx,Ry,Rz]) from the robot pose
    :return: euler angles in [rad]
    """

    temp = R.from_rotvec(orientation)
    euler = temp.as_euler("xyz", degrees=False)
    return np.array(euler)


def Euler2Robot(euler_angles):
    """
    Convert euler zyx angle to axis-angle
    :param: array of euler angles in xyz convention
    :return:  np.array([Rx,Ry,Rz])
    """
    temp2 = R.from_euler('xyz', euler_angles, degrees=False)
    axis_angles = temp2.as_rotvec()
    return np.array(axis_angles)


def Axis2Vector(axis_angles):
    """
    Convert axis-angle representation to the rotation vector form
    :param axis_angles: [Rx,Ry,Rz]
    :return: rot = [theta*ux,theta*uy,theta*uz] where:
    size is "theta"
    direction [ux,uy,uz] is a rotation vector
    """
    # axis_deg = np.rad2deg(axis_angles)
    size = np.linalg.norm(axis_angles)  # np.linalg.norm(axis_deg)
    direction = axis_angles / size  # axis_deg/size
    return size, direction


def Rot_matrix(angle, axis):
    if axis == 'x':
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    raise RuntimeError('!!Error in Rot_matrix(angle,axis): axis must take the values: "x","y",or "z" as characters!!')


def RotationVector(angles_current, angels_desired, angles_format='axis'):
    """
    Outputs: rotation vector "V21" in base frame. rotate from 1-curr to 2-desired.
    """
    if angles_format == 'axis':
        R01 = R.from_rotvec(angles_current).as_matrix()
        R10 = R01.T
        R02 = R.from_rotvec(angels_desired).as_matrix()
        R12 = np.dot(R10, R02)

        Vrot1 = R.from_matrix(R12).as_rotvec()
        Vrot0 = np.dot(R01, Vrot1)  # Vrot = R01*R12=R02 (in a vector form)
        return Vrot0

    elif angles_format == 'euler':
        R01 = R.from_euler('xyz', angles_current).as_matrix()
        R10 = R01.T
        R02 = R.from_euler('xyz', angels_desired).as_matrix()
        R12 = np.dot(R10, R02)

        Vrot1 = R.from_matrix(R12).as_rotvec()
        Vrot0 = np.dot(R01, Vrot1)
        return Vrot0

    print('!!Error in rotation_vector(): argument "angles_format" must get the values: "axis" or "euler"!!')
    return 'Error in rotation_vector()'


def Rotation_Matrix_To_Vector(intial_mat, final_mat):
    """
    EC
    Args: two rotation matrices in the same coordinate system representation
    Returns: axis angle representation of the Vrot_ref angle
    """
    R01 = intial_mat
    R10 = R01.T
    R02 = final_mat
    R12 = np.dot(R10, R02)

    Vrot1 = R.from_matrix(R12).as_rotvec()
    Vrot0 = np.dot(R01, Vrot1)  # Vrot = R01*R12=R02 (in a vector form)
    return Vrot0


def Vrot_to_axis(current_axis_angles, V_diff):
    """
    Args: current_axis_angles: orientation of the robot [Rx,Ry,Rz] V_diff: (Vrot_current-Vrot_ref): difference
    between rotation vector Vrot_current = atr.RotationVector(current_pose[3:6], desired_pose[3:6])-> rotation vector
    between start pose and desired pose. Vrot_ref given from the minimum-jerk trajectory, expresses rotation from
    reference fram to final desired.
    Returns: axis angle representation of the Vrot_ref angle
    """
    R01 = R.from_rotvec(current_axis_angles).as_matrix()
    R10 = R01.T
    V21 = np.dot(R10, V_diff)
    R12 = R.from_rotvec(V21).as_matrix()
    R21 = R12.T
    R20 = np.dot(R21, R10)
    R02 = R20.T
    angels_desired = R.from_matrix(R02).as_rotvec()  # V20 - the desired axis angles
    return angels_desired


def Rot_marix_to_axis_angles(Rot_matrix):
    Rotvec = R.from_matrix(Rot_matrix).as_rotvec()
    return Rotvec


def Gripper2Base_matrix(axis_angles_reading):
    R0t = R.from_rotvec(axis_angles_reading).as_matrix()
    return R0t


def Base2Gripper_matrix(axis_angles_reading):
    Rt0 = R.from_rotvec(axis_angles_reading).as_matrix().T
    return Rt0


def Tool2Base_vec(axis_angles_reading, vector):
    # "vector" is the vector that one would like to transform from Tool coordinate sys to the Base coordinate sys
    R0t = R.from_rotvec(axis_angles_reading).as_matrix()
    return R0t @ vector


def Base2Tool_vec(axis_angles_reading, vector):
    # "vector" is the vector that one would like to transform from Base coordinate sys to the Tool coordinate sys.
    Rt0 = (R.from_rotvec(axis_angles_reading).as_matrix()).T
    return Rt0 @ vector


def Tool2Base_multiple_vectors(axis_angles_reading, matrix):
    # "matrix" is matrix with all the vectors the one want to translate from Tool sys to Base sys.
    # matrix.shape should be nX3, when n is any real number of vectors.
    R0t = R.from_rotvec(axis_angles_reading).as_matrix()
    return (R0t @ (matrix.T)).T


def Base2Tool_multiple_vectors(axis_angles_reading, matrix):
    # "matrix" is matrix with all the vectors the one want to translate from Base sys to Tool sys.
    # matrix.shape should be nX3, when n is any real number of vectors
    Rt0 = (R.from_rotvec(axis_angles_reading).as_matrix()).T
    return (Rt0 @ (matrix.T)).T


def Base2Tool_sys_converting(coordinate_sys, pose_real, pose_ref, vel_real, vel_ref, F_internal, F_external,
                             force_reading):
    # "coordinate_sys" is the axis_angle vector which represent the rotation vector between Base sys to Tool sys.

    REAL_DATA_tool = Base2Tool_multiple_vectors(coordinate_sys, np.block(
        [[pose_real[:3]], [pose_real[3:]], [vel_real[:3]], [vel_real[3:]]]))
    [pose_real[:3], pose_real[3:], vel_real[:3], vel_real[3:]] = [REAL_DATA_tool[0], REAL_DATA_tool[1],
                                                                  REAL_DATA_tool[2], REAL_DATA_tool[3]]
    REF_DATA_tool = Base2Tool_multiple_vectors(coordinate_sys, np.block(
        [[pose_ref[:3]], [pose_ref[3:]], [vel_ref[:3]], [vel_ref[3:]]]))
    [pose_ref[:3], pose_ref[3:], vel_ref[:3], vel_ref[3:]] = [REF_DATA_tool[0], REF_DATA_tool[1], REF_DATA_tool[2],
                                                              REF_DATA_tool[3]]

    FORCE_DATA_tool = Base2Tool_multiple_vectors(coordinate_sys,
                                                 np.block([[force_reading[:3]], [F_internal[:3]], [F_external[:3]]]))
    [force_reading[:3], F_internal[:3], F_external[:3]] = [FORCE_DATA_tool[0], FORCE_DATA_tool[1], FORCE_DATA_tool[2]]

    MOMENT_DATA_tool = Base2Tool_multiple_vectors(coordinate_sys, np.block(
        [[force_reading[3:]], [F_internal[3:]], [F_external[3:]]]))
    [force_reading[3:], F_internal[3:], F_external[3:]] = [MOMENT_DATA_tool[0], MOMENT_DATA_tool[1],
                                                           MOMENT_DATA_tool[2]]
    return pose_real, pose_ref, vel_real, vel_ref, F_internal, F_external, force_reading
