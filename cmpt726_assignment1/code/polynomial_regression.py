#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()


def plot_error(normalized):
    targets = values[:, 1]
    if normalized=='yes':
        x = a1.normalize_data(values[:, 7:])
    else:
        x=values[:, 7:]
    N_TRAIN = 100
    x_train = x[0:N_TRAIN, :]
    x_test = x[N_TRAIN:, :]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]

    # Complete the linear_regression and evaluate_regression functions of the assignment1.py
    # Pass the required parameters to these functions

    tr_dicts = {}
    te_dicts = {}
    keys = range(1, 7)
    for degree in range(1, 7):
        (w, train_err) = a1.linear_regression(x_train, t_train, 'polynomial', 0, degree, 'yes', 0, 0)
        (t_est, test_err) = a1.evaluate_regression(x_test, t_test, w, degree, 'polynomial', 'yes', 0, 0)
        tr_dicts[degree] = float(train_err)
        te_dicts[degree] = float(test_err)

    # Produce a plot of results.
    plt.rcParams.update({'font.size': 15})
    plt.plot(list(tr_dicts.keys()), list(tr_dicts.values()))
    plt.plot(list(te_dicts.keys()), list(te_dicts.values()))
    plt.ylabel('RMS')
    plt.legend(['Training error', 'Testing error'])
    plt.title('Fit with polynomials, no regularization')
    plt.xlabel('Polynomial degree')
    plt.show()


#plot the curve for non-normalized input features
plot_error('no')
#plot the curve for normalized input feaatures
plot_error('yes')


