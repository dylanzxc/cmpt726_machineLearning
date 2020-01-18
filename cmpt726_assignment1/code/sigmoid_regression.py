#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
N_TRAIN = 100
targets = values[:, 1]
x_train = values[0:N_TRAIN, 10]
x_test = values[N_TRAIN:, 10]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

(w, train_err) = a1.linear_regression(x_train, t_train, 'sigmoid', 0, 0, 'none', [100, 10000], 2000.0)
print('training error:', train_err)
(t_est, test_err) = a1.evaluate_regression(x_test, t_test, w, 0, 'sigmoid', 'none', [100,10000], 2000.0)
print('testing error:', test_err)

x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
x_ev = np.asmatrix(x_ev)
phi_x_ev = a1.design_matrix('sigmoid', np.transpose(x_ev), 0, 'none', [100,10000], 2000.0)
y_ev = np.dot(phi_x_ev, w)
plt.plot(x_train,t_train,'yo')
plt.plot(x_test,t_test,'bo')
plt.plot(x_ev,np.transpose(y_ev),'r.-')
plt.legend(['Training data','Test data','Learned Polynomial'])
plt.title('Visualization of a function and some data points')
plt.show()
