import csv
import os

import seaborn as sns
import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

read_all = False
if read_all:
    """Code responsible for merging datasets"""
    df = pd.DataFrame()
    # loop through file names
    # for i in range(1, 101):
    for i in range(1, 8+1):
        file_name = "/home/user/Desktop/Simulation_n5/robosuite/data_collection/ep{}.csv".format(i)
        df_new = pd.read_csv(file_name)
        df = pd.concat([df, df_new], ignore_index=True)

    filename = "paired_errors.csv"
    filepath = os.path.join('/home/user/Desktop/Simulation_n5/robosuite/data_collection4', filename)
    df.to_csv(filepath)
    print('Done')

collect_positive = False
if collect_positive:
    """Creates new df only from the values that contains labels with overlap"""
    df = pd.DataFrame()
    for i in range(1, 8+1):
        file_name = "/home/user/Desktop/Simulation_n5/robosuite/data_collection4/ep{}.csv".format(i)
        df_new = pd.read_csv(file_name)
        if df_new.Case.sum() > 0: #and df_new.Fz.max() <= 80:
            # check if function has some overlap
            # option for trimming
            buffer = int(0.5 / 0.002)  # 4 sec of data
            length = len(df_new)
            first = df_new.Case.idxmax()
            last = df_new[::-1].Case.idxmax()
            if last+buffer<length:
                df2 = df_new.loc[first - buffer:last+buffer]
            else:
                df2 = df_new.loc[first - buffer:]
            # plt.figure()
            # plt.scatter(df2.t, df2.Case)
            # plt.show()
            # concat
            df = pd.concat([df, df2], ignore_index=True)
            # df = pd.concat([df, df_new], ignore_index=True)
        else:
            print(i)
            # plt.figure()
            # plt.scatter(df_new.t, df_new.Case)
            # plt.figure()
            # plt.plot(df_new.t, df_new.Fz)
            # plt.show()
            continue

    filename = "all_data_cut_0.5sec.csv"
    filepath = os.path.join('/home/user/Desktop/Simulation_n5/robosuite/data_collection4', filename)
    df.to_csv(filepath)
    print('Done')

collect_first_set_1 = False
if collect_first_set_1:
    """Creates new df only from the values that contains labels with first overlap"""
    df = pd.DataFrame()
    for i in range(1, 8+1):
        file_name = "/home/user/Desktop/Simulation_n5/robosuite/data_collection4/ep{}.csv".format(i)
        df_new = pd.read_csv(file_name)
        if df_new.Case.sum() > 0:
            buffer = int(1 / 0.002)  # 4 sec of data
            length = len(df_new)
            start_idx = df_new.index[df_new['Case'].diff() == 1].tolist()
            stop_idx = df_new.index[df_new['Case'].diff() == -1].tolist()

            df2 = df_new.iloc[start_idx[0]-buffer:stop_idx[0]+buffer]

            # plt.figure()
            # plt.scatter(df2.t, df2.Case)
            # plt.show()
            # concat
            df = pd.concat([df, df2], ignore_index=True)
            # df = pd.concat([df, df_new], ignore_index=True)
        else:
            print(i)
            # plt.figure()
            # plt.scatter(df_new.t, df_new.Case)
            # plt.figure()
            # plt.plot(df_new.t, df_new.Fz)
            # plt.show()
            continue

    filename = "all_data_cut_first_1sec.csv"
    filepath = os.path.join('/home/user/Desktop/Simulation_n5/robosuite/data_collection4', filename)
    df.to_csv(filepath)
    print('Done')


#
# 1063523
# 1077120
# 640097

# sample = pd.read_csv("/home/user/Desktop/Simulation_n5/robosuite/data_collection2/all_data_cut_2sec.csv")
# print(sample.shape)
# print(sample.Case.sum())
# # first only use those that have labels
# buffer = int(4/0.002) # 4 sec of data
# length = len(sample)
# first = sample.case.idxmax()
# last = sample[::-1].case.idxmax()
# new_df = sample.loc[first-buffer:]
#
#

#
# sample = pd.read_csv("/home/user/Desktop/Simulation_n5/robosuite/data_collection4/ep1.csv")
# # #
# plt.figure()
# plt.plot(sample.t, sample.Fz)
#
# plt.figure()
# plt.scatter(sample.t, sample.Case)
# plt.show()