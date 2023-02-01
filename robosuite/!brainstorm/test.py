
import pandas as pd
import numpy as np
from copy import deepcopy

df = pd.read_csv('/home/user/Desktop/ML/robotdatacollection3/merged_robot_data_100episodes.csv')
df = df.drop(['Unnamed: 0.1','Unnamed: 0','t'], axis=1)
feature_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My']
# feature_names = ['fz']
#
# feature_names = ['fx', 'fy', 'fz', 'mx', 'my']

X_df = df[feature_names]
X = X_df.to_numpy()
y = df.Case.to_numpy()

window_len = 200#300

if window_len:
    n_features = len(feature_names)  # 3

    row = X.shape[0] + 1 - window_len
    col_len = n_features * window_len
    new_x = np.zeros((row, col_len))
    new_y = np.zeros((row, 1))

    for i in range(len(new_x)):
        new_x[i] = X[i:i + window_len].reshape(1, col_len)[0][::-1]
        new_y[i] = y[i + (window_len - 1)]

    y = deepcopy(new_y)
    X = deepcopy(new_x)

print('Done')