#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
#x = a1.normalize_data(x)

N_TRAIN = 100
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

tr_dicts={}
te_dicts={}


def plot_multiple_error(title,bias):
    #for features 8-15
    for iterator in range(7, 15):
        x = values[:, iterator]
        x_train = x[0:N_TRAIN, :]
        x_test = x[N_TRAIN:, :]
        (w, train_err) = a1.linear_regression(x_train, t_train, 'polynomial', 0, 3, bias, 0, 0)
        (t_est, test_err) = a1.evaluate_regression(x_test, t_test, w, 3, 'polynomial', bias, 0, 0)
        #store the values in two dicts
        tr_dicts[1 + iterator] = float(train_err)
        te_dicts[1.35 + iterator] = float(test_err)
    #print(tr_dicts)
    #print(te_dicts)

    # Produce a plot of results.
    plt.rcParams.update({'font.size': 15})
    plt.bar(list(tr_dicts.keys()), list(tr_dicts.values()), width=0.35)
    plt.bar(list(te_dicts.keys()), list(te_dicts.values()), width=0.35)
    plt.ylabel('RMS')
    plt.legend(['Training error', 'Testing error'])
    plt.title('Fit with degree=3 polynomial, no regularization,'+ title)
    plt.xlabel('Feature index')
    plt.show()


plot_multiple_error('without bias','none')
plot_multiple_error('with bias','yes')
