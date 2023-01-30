import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

"""
case=1
    overlap big enough to perform impedance
case=2
    overlap not big enough overlap for impedance
case=3
    no overlap
"""

df1 = pd.read_csv("test1.csv")
df2 = pd.read_csv("test2.csv")
df3 = pd.read_csv("test3.csv")
df4 = pd.read_csv("test4.csv")
df5 = pd.read_csv("test5.csv")
df6 = pd.read_csv("test6.csv")
df7 = pd.read_csv("test7.csv")
df8 = pd.read_csv("test8.csv")


contact_time1 = df1['t_contact'][0]
contact_time2 = df2['t_contact'][0]
contact_time3 = df3['t_contact'][0]
contact_time4 = df4['t_contact'][0]
contact_time5 = df5['t_contact'][0]
contact_time6 = df6['t_contact'][0]
contact_time7 = df7['t_contact'][0]
contact_time8 = df8['t_contact'][0]


df11 = df1[df1['time'] > contact_time1]
df22 = df2[df2['time'] > contact_time2]
df33 = df3[df3['time'] > contact_time3]
df44 = df4[df4['time'] > contact_time4]
df55 = df5[df5['time'] > contact_time5]
df66 = df6[df6['time'] > contact_time6]
df77 = df7[df7['time'] > contact_time7]
df88 = df8[df8['time'] > contact_time8]


frames = [df11, df22, df33, df44, df55, df66, df77, df88]
result = pd.concat(frames)

# adding moment magnitude
df11['m_mag'] = np.sqrt(df11['Mx']**2+df11['My']**2)
result['m_mag'] = np.sqrt(result['Mx']**2+result['My']**2)

impedance = result[result['Case'] == 1]
overlap = result[result['Case'] == 2]
no_overlap = result[result['Case'] == 3]

impedance_one = df11[df11['Case'] == 1]
overlap_one = df11[df11['Case'] == 2]
no_overlap_one = df11[df11['Case'] == 3]

#
# plt.figure()
# plt.plot(df11['time'],df11['Fz'], label='Fz_sensor')
# plt.grid()
# plt.xlabel('Time[sec]')
# plt.ylabel('Fz[N]')
#
# plt.figure()
# plt.plot(df11['time'],df11['Mx'], label='Mx_sensor')
# plt.grid()
# plt.xlabel('Time[sec]')
# plt.ylabel('Mx[Nm]')
#
# plt.figure()
# plt.plot(df11['time'],df11['My'], label='My_sensor')
# plt.grid()
# plt.xlabel('Time[sec]')
# plt.ylabel('My[Nm]')
#
# plt.figure("Zones")
# plt.scatter(df11['time'], df11['Case'])
# plt.xlabel('Time [sec]')
# plt.show()

# # TODO Single case analysis of Fz
# plt.figure("Force in z direction")
# ax1 = plt.subplot(311)
# ax1.plot(np.arange(0, impedance_one.shape[0], 1), impedance_one['Fz'], 'b', label='Impedance')
# ax1.legend()
# ax1.set_title('Case1: possible insertion using impedance')
#
# ax2 = plt.subplot(312)
# ax2.plot(np.arange(0, overlap_one.shape[0], 1), overlap_one['Fz'], 'b', label='Overlap')
# ax2.legend()
# ax2.set_title('Case2: Overlap but not possible insertion')
#
# ax3 = plt.subplot(313)
# ax3.plot(np.arange(0, no_overlap_one.shape[0], 1), no_overlap_one['Fz'], 'b', label='No Overlap')
# ax3.legend()
# ax3.set_title('Case3: No Overlap')
# plt.show()

# TODO single case analysis of Mx

# plt.figure("Moment in x direction")
# ax1 = plt.subplot(311)
# ax1.plot(np.arange(0, impedance_one.shape[0], 1), impedance_one['Mx'], 'b', label='Impedance')
# ax1.legend()
# ax1.set_title('Case1: possible insertion using impedance')
#
# ax2 = plt.subplot(312)
# ax2.plot(np.arange(0, overlap_one.shape[0], 1), overlap_one['Mx'], 'b', label='Overlap')
# ax2.legend()
# ax2.set_title('Case2: Overlap but not possible insertion')
#
# ax3 = plt.subplot(313)
# ax3.plot(np.arange(0, no_overlap_one.shape[0], 1), no_overlap_one['Mx'], 'b', label='No Overlap')
# ax3.legend()
# ax3.set_title('Case3: No Overlap')
# plt.show()

# TODO single case analysis of My

# plt.figure("Moment in y direction")
# ax1 = plt.subplot(311)
# ax1.plot(np.arange(0, impedance_one.shape[0], 1), impedance_one['My'], 'b', label='Impedance')
# ax1.legend()
# ax1.set_title('Case1: possible insertion using impedance')
#
# ax2 = plt.subplot(312)
# ax2.plot(np.arange(0, overlap_one.shape[0], 1), overlap_one['My'], 'b', label='Overlap')
# ax2.legend()
# ax2.set_title('Case2: Overlap but not possible insertion')
#
# ax3 = plt.subplot(313)
# ax3.plot(np.arange(0, no_overlap_one.shape[0], 1), no_overlap_one['My'], 'b', label='No Overlap')
# ax3.legend()
# ax3.set_title('Case3: No Overlap')
# plt.show()

# TODO histogram of Fz
#
# plt.figure()
# sns.histplot(data=df11, x="Fz", hue='Case', kde=True, palette=["blue", "r", "C2"])
#
# fig, axes = plt.subplots(3, 1, figsize=(10, 15))
# sns.histplot(ax=axes[0], data=impedance_one, x="Fz", kde=True, color='blue')
# axes[0].set_title('Case1: possible insertion using impedance')
# sns.histplot(ax=axes[1], data=overlap_one, x="Fz", kde=True, color='red')
# axes[1].set_title('Case2: Overlap but not possible insertion')
# sns.histplot(ax=axes[2], data=no_overlap_one, x="Fz", kde=True, color='C2')
# axes[2].set_title('Case3: No Overlap')
# plt.show()

# TODO magnitude of moments vs time and histograms
# plt.figure('Magnitude of Moments vs Time')
# plt.plot(df11['time'], df11['m_mag'], label='|M|')
# plt.grid()
# plt.xlabel('Time[sec]')
# plt.ylabel('|M|')
# plt.legend()
#
# plt.figure()
# sns.histplot(data=df11, x="m_mag", hue='Case', kde=True, palette=["blue", "r", "C2"])
#
# fig, axes = plt.subplots(3, 1, figsize=(10, 15))
# sns.histplot(ax=axes[0], data=impedance_one, x="m_mag", kde=True, color='blue')
# axes[0].set_title('Case1: possible insertion using impedance')
# sns.histplot(ax=axes[1], data=overlap_one, x="m_mag", kde=True, color='red')
# axes[1].set_title('Case2: Overlap but not possible insertion')
# sns.histplot(ax=axes[2], data=no_overlap_one, x="m_mag", kde=True, color='C2')
# axes[2].set_title('Case3: No Overlap')
# plt.show()

# TODO 2d plot of mag vs fz
# c_dict = {1: 'blue', 2: 'red', 3: 'C2'}
# c_dict2 = {1: 'blue', 2: 'red', 3: 'none'}
# scatter_x = df11['Fz'].values
# scatter_y = df11['m_mag'].values
# group = df11['Case'].values
#
# fig, ax = plt.subplots()
# for g in np.unique(group):
#     ix = np.where(group == g)
#     ax.scatter(scatter_x[ix], scatter_y[ix], label=g, s=20, facecolors=c_dict2[g], edgecolors=c_dict[g])
# ax.set_title('Plot: Fz vs |M|')
# ax.set_xlabel('Fz')
# ax.set_ylabel('|M|')
# ax.legend()
#
# f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# ax1.scatter(impedance_one['Fz'].values, impedance_one['m_mag'].values, label='1', s=20, facecolors='None', edgecolors='blue')
# ax1.legend()
# ax1.set_ylabel('|M|')
# ax2.scatter(overlap_one['Fz'].values, overlap_one['m_mag'].values, label='2', s=20, facecolors='None', edgecolors='red')
# ax2.legend()
# ax2.set_ylabel('|M|')
# ax3.scatter(no_overlap_one['Fz'].values, no_overlap_one['m_mag'].values, label='3', s=20, facecolors='None', edgecolors='green')
# ax3.set_xlabel('Fz')
# ax3.set_ylabel('|M|')
# ax3.legend()
# plt.show()

# TODO 3 d plot
# a = df11['Fz'].values
# labels = df11['Case'].values
# # a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
# # labels = np.array([1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 1, 2, 3])
# i = 0
# b = np.zeros((len(a)-2, 3))
# bb = np.zeros((len(a)-2, 1))
# while i + 3 <= len(a):
#     b[i] = a[i:i+3]
#     bb[i] = labels[i+2]
#     i += 1


# c_dict = {1: 'blue', 2: 'red', 3: 'C2'}
# scatter_x = b[:, 0]
# scatter_y = b[:, 1]
# scatter_z = b[:, 2]
# group = bb[:, 0]
#
# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(111, projection='3d')
# for g in np.unique(group):
#     ix = np.where(group == g)
#     ax.scatter(scatter_x[ix], scatter_y[ix], scatter_z[ix], label=g, s=20, facecolors=None, edgecolors=c_dict[g])
# ax.set_xlabel('$Fz_{t}$')
# ax.set_ylabel('$Fz_{t+1}$')
# ax.set_zlabel('$Fz_{t+2}$')
# ax.legend()
# plt.show()



# TODO  Analysis of all 8 cases
# plt.figure()
# sns.histplot(data=result, x="Fz", hue='Case', kde=True,  palette=["blue", "r", "C2"])
#
# fig, axes = plt.subplots(3, 1, figsize=(10, 15))
# sns.histplot(ax=axes[0], data=impedance, x="Fz", kde=True, color='blue')
# axes[0].set_title('Case1: possible insertion using impedance - 8 measurements')
# sns.histplot(ax=axes[1], data=overlap, x="Fz", kde=True, color='red')
# axes[1].set_title('Case2: Overlap but not possible insertion- 8 measurements')
# sns.histplot(ax=axes[2], data=no_overlap, x="Fz", kde=True, color='C2')
# axes[2].set_title('Case3: No Overlap - 8 measurements')
# plt.show()

# TODO histogram moments (8 experiments)

#
# plt.figure()
# sns.histplot(data=result, x="m_mag", hue='Case', kde=True, palette=["blue", "r", "C2"])
#
# fig, axes = plt.subplots(3, 1, figsize=(10, 15))
# sns.histplot(ax=axes[0], data=impedance, x="m_mag", kde=True, color='blue')
# axes[0].set_title('Case1: possible insertion using impedance')
# sns.histplot(ax=axes[1], data=overlap, x="m_mag", kde=True, color='red')
# axes[1].set_title('Case2: Overlap but not possible insertion')
# sns.histplot(ax=axes[2], data=no_overlap, x="m_mag", kde=True, color='C2')
# axes[2].set_title('Case3: No Overlap')
# plt.show()

# TODO 2d plot of mag vs fz (8 experiments)
# c_dict = {1: 'blue', 2: 'red', 3: 'C2'}
# c_dict2 = {1: 'blue', 2: 'red', 3: 'none'}
# scatter_x = result['Fz'].values
# scatter_y = result['m_mag'].values
# group = result['Case'].values
#
# fig, ax = plt.subplots()
# for g in np.unique(group):
#     ix = np.where(group == g)
#     ax.scatter(scatter_x[ix], scatter_y[ix], label=g, s=20, facecolors=c_dict2[g], edgecolors=c_dict[g])
# ax.set_title('Plot: Fz vs |M|')
# ax.set_xlabel('Fz')
# ax.set_ylabel('|M|')
# ax.legend()
#
# f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# ax1.scatter(impedance['Fz'].values, impedance['m_mag'].values, label='1', s=20, facecolors='None', edgecolors='blue')
# ax1.legend()
# ax1.set_ylabel('|M|')
# ax2.scatter(overlap['Fz'].values, overlap['m_mag'].values, label='2', s=20, facecolors='None', edgecolors='red')
# ax2.legend()
# ax2.set_ylabel('|M|')
# ax3.scatter(no_overlap['Fz'].values, no_overlap['m_mag'].values, label='3', s=20, facecolors='None', edgecolors='green')
# ax3.set_xlabel('Fz')
# ax3.set_ylabel('|M|')
# ax3.legend()
# plt.show()

# # TODO 3 d plot (8 experiments)
# a = result['Fz'].values
# labels = result['Case'].values
# # a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
# # labels = np.array([1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 1, 2, 3])
# i = 0
# b = np.zeros((len(a)-2, 3))
# bb = np.zeros((len(a)-2, 1))
# while i + 3 <= len(a):
#     b[i] = a[i:i+3]
#     bb[i] = labels[i+2]
#     i += 1
#
#
# c_dict = {1: 'blue', 2: 'red', 3: 'C2'}
# scatter_x = b[:, 0]
# scatter_y = b[:, 1]
# scatter_z = b[:, 2]
# group = bb[:, 0]
#
# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(111, projection='3d')
# for g in np.unique(group):
#     ix = np.where(group == g)
#     ax.scatter(scatter_x[ix], scatter_y[ix], scatter_z[ix], label=g, s=20, facecolors=None, edgecolors=c_dict[g])
# ax.set_xlabel('$Fz_{t}$')
# ax.set_ylabel('$Fz_{t+1}$')
# ax.set_zlabel('$Fz_{t+2}$')
# ax.legend()
# plt.show()
