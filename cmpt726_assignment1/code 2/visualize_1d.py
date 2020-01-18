#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
N_TRAIN = 100
targets = values[:,1]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
def plot_fit(feature,bias,title,linspace_scale):
    x_train=values[0:N_TRAIN,feature-1]
    x_test=values[N_TRAIN:,feature-1]
    (w, train_err) = a1.linear_regression(x_train, t_train, 'polynomial', 0, 3, bias, 0, 0)
    if linspace_scale=='small':
        x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
    if linspace_scale=='large':
        x_ev = np.linspace(np.asscalar(min(min(x_train), min(x_test))),
                           np.asscalar(max(max(x_train), max(x_test))), num=500)
    x_ev=np.asmatrix(x_ev)
    phi_x_ev=a1.design_matrix('polynomial',np.transpose(x_ev),3,bias,0,0)
    y_ev=np.dot(phi_x_ev,w)
    plt.plot(x_train,t_train,'yo')
    plt.plot(x_test,t_test,'bo')
    plt.plot(x_ev,np.transpose(y_ev),'r.-')
    plt.legend(['Training data','Test data','Learned Polynomial'])
    plt.title('Visualization of a function and some data points'+title)
    plt.show()


plot_fit(11,'yes', 'for feature11 with bias', 'small')
plot_fit(11,'yes', 'for feature11 with bias', 'large')
plot_fit(11,'none', 'for feature11 without bias','small')
plot_fit(11,'none', 'for feature11 without bias', 'large')
plot_fit(12,'yes','for feature12 with bias','small')
plot_fit(12,'none','for feature12 without bias','small')
plot_fit(13,'yes','for feature13 with bias','small')
plot_fit(13,'none', 'for feature13 without bias','small')
