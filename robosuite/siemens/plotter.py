import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv('/home/danieln1/Downloads/original_n5/robosuite/siemens/data_daniel.csv')
# df = pd.read_csv('/home/danieln1/Downloads/original_n5/robosuite/siemens/data.csv')
print(df.columns)
t = df['time']

plt.figure()
ax1 = plt.subplot(311)
ax1.plot(t, df['Xm position'], 'g--', label='Xm position')
ax1.plot(t, df['Xr position'], 'b', label='Xr position')
ax1.plot(t, df['X_ref position'], 'r--', label='X_ref position')
ax1.legend()
ax1.set_title('X Position [m]')

ax2 = plt.subplot(312)
ax2.plot(t, df['Ym position'], 'g--', label='Ym position')
ax2.plot(t, df['Yr position'], 'b', label='Yr position')
ax2.plot(t, df['Y_ref position'], 'r--', label='Y_ref position')
ax2.legend()
ax2.set_title('Y Position [m]')

ax3 = plt.subplot(313)
ax3.plot(t, df['Zm position'], 'g--', label='Zm position')
ax3.plot(t, df['Zr position'], 'b', label='Zr position')
ax3.plot(t, df['Z_ref position'], 'r--', label='Z_ref position')
ax3.legend()
ax3.set_title('Z Position [m]')

plt.figure()
ax1 = plt.subplot(311)
ax1.plot(t, df['Fx'], 'b', label='Fx')
ax1.plot(t, df['Fx_des'], 'g', label='Fx_des')
ax1.legend()
ax1.set_title('Fx [N]')

ax2 = plt.subplot(312)
ax2.plot(t, df['Fy'], 'b', label='Fy')
ax2.plot(t, df['Fy_des'], 'g', label='Fy_des')
ax2.legend()
ax2.set_title('Fy [N]')

ax3 = plt.subplot(313)
ax3.plot(t, df['Fz'], 'b', label='Fz')
ax3.plot(t, df['Fz_des'], 'g', label='Fz_des')
ax3.legend()
ax3.set_title('Fz [N]')

plt.figure()
ax1 = plt.subplot(311)
ax1.plot(t, df['Mx'], 'b', label='Mx')
ax1.plot(t, df['mx_des'], 'g', label='mx_des')
ax1.legend()
ax1.set_title('Mx [Nm]')

ax2 = plt.subplot(312)
ax2.plot(t, df['My'], 'b', label='My')
ax2.plot(t, df['my_des'], 'g', label='My_des')
ax2.legend()
ax2.set_title('My [Nm]')

ax3 = plt.subplot(313)
ax3.plot(t, df['Mz'], 'b', label='Mz')
ax3.plot(t, df['mz_des'], 'g', label='Mz_des')
ax3.legend()
ax3.set_title('Mz [Nm]')

plt.show()

