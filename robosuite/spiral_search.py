import numpy as np
from matplotlib import pyplot as plt

r_peg = 0.0021
r_hole = 0.0024
clearance = r_hole - r_peg
# print(clearance)
# 0.0003
# p<=2*clearance

def next_spiral(theta_current, radius_current):
    v = 0.000153
    p = 0.0002  # set it to clearance value
    dt = 0.002
    theta_dot_current = (2*np.pi*v)/(p*np.sqrt(1+theta_current))

    theta_next = theta_current + theta_dot_current * dt
    radius_next = (p/(2*np.pi))*theta_next

    x_next = radius_next * np.cos(theta_next)
    y_next = radius_next * np.sin(theta_next)

    return theta_next, radius_next, x_next, y_next

if __name__ == "__main__":
    x_array = [0]
    y_array = [0]
    for i in range(100000):
        if i < 1:
            theta_init = 0
            radius_init = 0
        theta_next, radius_next, x_next, y_next = next_spiral(theta_init, radius_init)
        theta_init = np.copy(theta_next)
        radius_init = np.copy(radius_next)

        x_array.append(x_next)
        y_array.append(y_next)

plt.figure()
plt.plot(x_array, y_array)
plt.axhline(y=0, color='r', linestyle='--')
plt.axvline(x=0, color='r', linestyle='--')
plt.show()