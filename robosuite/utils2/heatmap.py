import seaborn as sns
import random
import numpy as np
from matplotlib import pyplot as plt


import pickle
file = '/home/user/Desktop/Simulation_n5/robosuite/daniel_n8_sim/sim15_n8/scaled_error'
name = file + '.pkl'
with open(name, 'rb') as f:
    loaded_dict = pickle.load(f)
print()
success = np.array(loaded_dict.get('success'))
errors = np.array(loaded_dict.get('error'))

peg_radius = 2.1/1000
hole_radius = 2.4/1000

r_low = 0.6/1000
r_high = 0.8/1000
R_hole = hole_radius
R_peg = peg_radius
theta = np.linspace(0, 2*np.pi, 1000)
x_hole = R_hole*np.cos(theta)
y_hole = R_hole*np.sin(theta)
x_peg = R_peg*np.cos(theta)
y_peg = R_peg
x_low = r_low*np.cos(theta)
y_low = r_low*np.sin(theta)
x_high = r_high*np.cos(theta)
y_high = r_high*np.sin(theta)


plt.figure()
# plt.title(file + ': 100 eval ep: ring error: [0.6, 0.8]')
plt.plot(x_hole, y_hole, color='k', label='hole', linewidth=3)
plt.plot(x_low, y_low, color='r', linestyle='dashed')
plt.plot(x_high, y_high, color='r', linestyle='--')
plt.axhline(y=0, color='k', linestyle='dotted')
plt.axvline(x=0, color='k', linestyle='dotted')
for i in range(len(errors)):
    err = errors[i]
    suc = success[i]
    if suc:
        plt.scatter(err[0], err[1], c='green')
    else:
        plt.scatter(err[0], err[1], c='red')
        # print(f'Unsuccessful insertion for: {err[:2]*1000}mm')

plt.grid()
plt.show()



# plt.figure()
# plt.title(': 100 eval ep: ring error: [0.6, 0.8]')
# plt.plot(x_hole, y_hole, color='k', label='hole', linewidth=3)
# plt.plot(x_low, y_low, color='r', linestyle='dashed')
# plt.plot(x_high, y_high, color='r', linestyle='--')
# plt.axhline(y=0, color='k', linestyle='dotted')
# plt.axvline(x=0, color='k', linestyle='dotted')
# for i in range(100):
#     r = random.uniform(r_low, r_high)
#     # print('radial error in [mm]', r*1000)
#     theta = random.uniform(np.pi, 1.5 * np.pi)
#     x_error = r * np.cos(theta)
#     y_error = r * np.sin(theta)
#     plt.scatter(x_error, y_error, c='b')
#
# plt.grid()
# plt.show()