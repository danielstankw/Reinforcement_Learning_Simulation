import numpy as np
import pandas as pd


df = pd.read_csv('no_ml_x4.0.csv')
a = np.array(df)
sim_times = [float(x) for x in a[0][1][1:-1].split(',')]
average = np.mean(sim_times)
print(np.round(average,10))