import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
df = pd.read_csv("data.csv")

t = df['time']
# impedance
impedance_model_pos_vec_x = df['Xm position']
impedance_model_pos_vec_y = df['Ym position']
impedance_model_pos_vec_z = df['Zm position']
# robot pose
ee_pos_vec_x = df['Xr position']
ee_pos_vec_y = df['Yr position']
ee_pos_vec_z = df['Zr position']
# minimum jerk
pos_min_jerk_x = df['X_ref position']
pos_min_jerk_y = df['Y_ref position']
pos_min_jerk_z = df['Z_ref position']
# forces
wrench_vec_int_Fx = df['Fx']
wrench_vec_int_Fy = df['Fy']
wrench_vec_int_Fz = df['Fz']
desired_force_x = df['Fx_des']
desired_force_y = df['Fy_des']
desired_force_z = df['Fz_des']
# moments
wrench_vec_int_Mx = df['Mx']
wrench_vec_int_My = df['My']
wrench_vec_int_Mz = df['Mz']
desired_torque_x = df['mx_des']
desired_torque_y = df['my_des']
desired_torque_z = df['mz_des']
# ################################################################################################################
# idx = int(6 / 0.002)
# idx2 = int(7.3 / 0.002)

plt.figure()
ax1 = plt.subplot(311)
ax1.plot(t, impedance_model_pos_vec_x, 'g--', label='Xm position')
ax1.plot(t, ee_pos_vec_x, 'b', label='Xr position')
ax1.plot(t, pos_min_jerk_x, 'r--', label='X_ref position')
ax1.legend()
ax1.set_title('X Position [m]')

ax2 = plt.subplot(312)
ax2.plot(t, impedance_model_pos_vec_y, 'g--', label='Ym position')
ax2.plot(t, ee_pos_vec_y, 'b', label='Yr position')
ax2.plot(t, pos_min_jerk_y, 'r--', label='Y_ref position')
ax2.legend()
ax2.set_title('Y Position [m]')

ax3 = plt.subplot(313)
ax3.plot(t, impedance_model_pos_vec_z, 'g--', label='Zm position')
ax3.plot(t, ee_pos_vec_z, 'b', label='Zr position')
ax3.plot(t, pos_min_jerk_z, 'r--', label='Z_ref position')
ax3.legend()
ax3.set_title('Z Position [m]')
################################################################################################################
plt.figure()
ax1 = plt.subplot(311)
ax1.plot(t, wrench_vec_int_Fx, 'b', label='Fx')
ax1.plot(t, desired_force_x, 'g', label='Fx_des')
ax1.legend()
ax1.set_title('Fx [N]')

ax2 = plt.subplot(312)
ax2.plot(t, wrench_vec_int_Fy, 'b', label='Fy')
ax2.plot(t, desired_force_y, 'g', label='Fy_des')
ax2.legend()
ax2.set_title('Fy [N]')

ax3 = plt.subplot(313)
ax3.plot(t, wrench_vec_int_Fz, 'b', label='Fz')
ax3.plot(t, desired_force_z, 'g', label='Fz_des')
ax3.legend()
ax3.set_title('Fz [N]')
################################################################################################################
plt.figure()
ax1 = plt.subplot(311)
ax1.plot(t, wrench_vec_int_Mx, 'b', label='Mx')
ax1.plot(t, desired_torque_x, 'g', label='mx_des')
ax1.legend()
ax1.set_title('Mx [Nm]')

ax2 = plt.subplot(312)
ax2.plot(t, wrench_vec_int_My, 'b', label='My')
ax2.plot(t, desired_torque_y, 'g', label='My_des')
ax2.legend()
ax2.set_title('My [Nm]')

ax3 = plt.subplot(313)
ax3.plot(t, wrench_vec_int_Mz, 'b', label='Mz')
ax3.plot(t, desired_torque_z, 'g', label='Mz_des')
ax3.legend()
ax3.set_title('Mz [Nm]')
plt.show()
