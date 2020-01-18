#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100
targets = values[:N_TRAIN,1]
x = x[0:N_TRAIN,:]
lambda_list=[0, 0.01, 0.1, 1, 10, 100, 1000, 10000]
average_list = []

for i in [0, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
    sum = 0
    for fold in range(1, 11):
        x_vali = x[(fold - 1) * 10:fold * 10, :]
        t_vali = targets[(fold - 1) * 10:fold * 10]
        x_train = np.vstack((x[0:(fold - 1) * 10, :], x[10 * fold:, :]))
        t_train = np.vstack((targets[0:(fold - 1) * 10], targets[10 * fold:]))
        (w, train_err) = a1.linear_regression(x_train, t_train, 'polynomial', i, 2, 'yes', 0, 0)
        (t_est, test_err) = a1.evaluate_regression(x_vali, t_vali, w, 2, 'polynomial', 'yes', 0, 0)
        #print(test_err)
        sum=sum+float(test_err)
        #print(sum)
    average=float(sum/10)
    print(average)
    average_list.append(average)




plt.rcParams.update({'font.size': 15})
plt.semilogx(lambda_list,average_list)
plt.ylabel('Average RMS')
plt.legend(['Average Validation error'])
plt.title('Fit with degree 2 polynomials')
plt.xlabel('log scale lambda')
plt.show()


