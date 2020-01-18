#!/usr/bin/env python

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2

tol = 0.00001
max_iter = 500
etas = [0.5, 0.3, 0.1, 0.05, 0.01]
data = np.genfromtxt('data.txt')
legend=[]
np.random.shuffle(data)
X = data[:, 0:3]
t = data[:, 3]

for eta in etas:
    w = np.array([0.1, 0, 0])
    e_all = []
    legend.append(str(eta))
    for itr in range(0, max_iter):
        for i in range(0, len(X)):
            y = sps.expit(np.dot(X[i], w))
            grad_e = np.multiply((y - t[i]), X[i,:].T)
            w = w - eta * grad_e

        y = sps.expit(np.dot(X, w))
        e = -np.mean(np.multiply(t, np.log(y+1e-5)) + np.multiply((1 - t), np.log(1 - y+1e-5)))
        e_all.append(e)
        if itr > 0:
            if np.absolute(e_all[itr] - e_all[itr - 1]) < tol:
                break
    plt.plot(e_all)

plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.legend(legend)
plt.show()



