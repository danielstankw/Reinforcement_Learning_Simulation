import numpy as np

my_data = np.genfromtxt('action_matrix.csv', delimiter=',')
print(my_data.shape)

dims = 26

if dims == 26:
    m = np.mean(my_data, axis=0)
    std = np.std(my_data, axis=0)
    # print(m)
    # print(std)

    K_mean_mat = np.array([[abs(m[0]), 0, 0, 0, m[1], 0],
                  [0, abs(m[2]), 0, m[3], 0, 0],
                  [0, 0, abs(m[4]), 0, 0, 0],
                  [0, m[5], 0, abs(m[6]), 0, 0],
                  [m[7], 0, 0, 0, abs(m[8]), 0],
                  [0, 0, 0, 0, 0, abs(m[9])]])

    C_mean_mat = np.array([[abs(m[10]), 0, 0, 0, m[11], 0],
                  [0, abs(m[12]), 0, m[13], 0, 0],
                  [0, 0, abs(m[14]), 0, 0, 0],
                  [0, m[15], 0, abs(m[16]), 0, 0],
                  [m[17], 0, 0, 0, abs(m[18]), 0],
                  [0, 0, 0, 0, 0, abs(m[19])]])

    M_mean_mat = np.array([[abs(m[20]), 0, 0, 0, 0, 0],
                  [0, abs(m[21]), 0, 0, 0, 0],
                  [0, 0, abs(m[22]), 0, 0, 0],
                  [0, 0, 0, abs(m[23]), 0, 0],
                  [0, 0, 0, 0, abs(m[24]), 0],
                  [0, 0, 0, 0, 0, abs(m[25])]])
    print('K mean matrix')
    print(K_mean_mat)
    print('C mean matrix')
    print(C_mean_mat)
    print('M mean matrix')
    print(M_mean_mat)

    K_std_mat = np.array([[abs(std[0]), 0, 0, 0, std[1], 0],
                           [0, abs(std[2]), 0, std[3], 0, 0],
                           [0, 0, abs(std[4]), 0, 0, 0],
                           [0, m[5], 0, abs(std[6]), 0, 0],
                           [m[7], 0, 0, 0, abs(std[8]), 0],
                           [0, 0, 0, 0, 0, abs(std[9])]])

    C_std_mat = np.array([[abs(std[10]), 0, 0, 0, std[11], 0],
                           [0, abs(std[12]), 0, std[13], 0, 0],
                           [0, 0, abs(std[14]), 0, 0, 0],
                           [0, std[15], 0, abs(std[16]), 0, 0],
                           [std[17], 0, 0, 0, abs(std[18]), 0],
                           [0, 0, 0, 0, 0, abs(std[19])]])

    M_std_mat = np.array([[abs(std[20]), 0, 0, 0, 0, 0],
                           [0, abs(std[21]), 0, 0, 0, 0],
                           [0, 0, abs(std[22]), 0, 0, 0],
                           [0, 0, 0, abs(std[23]), 0, 0],
                           [0, 0, 0, 0, abs(std[24]), 0],
                           [0, 0, 0, 0, 0, abs(std[25])]])
    print('K std matrix')
    print(K_std_mat)
    print('C std matrix')
    print(C_std_mat)
    print('M std matrix')
    print(M_std_mat)

    Cv_k = np.divide(K_std_mat, K_mean_mat)
    Cv_c = np.divide(C_std_mat, C_mean_mat)
    Cv_m = np.divide(M_std_mat, M_mean_mat)
    print('Coefficient of variation')
    print(Cv_k)
    print(Cv_c)
    print(Cv_m)
    print('hi')
if dims == 18:
    m = np.mean(my_data, axis=0)
    std = np.std(my_data, axis=0)
    print(m)
    print(std)

    K_mean_mat = np.diag(m[:6])
    C_mean_mat = np.diag(m[6:12])
    M_mean_mat = np.diag(m[12:])
    print('K mean matrix')
    print(K_mean_mat)
    print('C mean matrix')
    print(C_mean_mat)
    print('M mean matrix')
    print(M_mean_mat)

    K_std_mat = np.diag(std[:6])
    C_std_mat = np.diag(std[6:12])
    M_std_mat = np.diag(std[12:])
    print('K std matrix')
    print(K_std_mat)
    print('C std matrix')
    print(C_std_mat)
    print('M std matrix')
    print(M_std_mat)

    Cv_k = np.divide(K_std_mat, K_mean_mat)
    Cv_c = np.divide(C_std_mat, C_mean_mat)
    Cv_m = np.divide(M_std_mat, M_mean_mat)
    print('Coefficient of variation')
    print(Cv_k)
    print(Cv_c)
    print(Cv_m)
    print('hi')
