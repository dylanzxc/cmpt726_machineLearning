"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
from scipy import nanmean

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding='latin1')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    
    return (x - mvec)/stdvec
    

def linear_regression(x, t, basis, reg_lambda, degree, bias, mu, s):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      bias is string, declare if we need bias term or not
      mu,s are parameters of sigmoid basis

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # Construct the design matrix.
    # Pass the required parameters to this function
    phi = design_matrix(basis, x, degree, bias, mu, s)
    pseudo_inverse = np.linalg.pinv(phi)
    # Learning Coefficients

    # Construct the weights for regularized model
    if reg_lambda > 0:
        # regularized regression
        n=(np.size(x,1))*degree+1
        I=np.identity(n)
        inv=np.linalg.inv(I*reg_lambda+np.dot(np.transpose(phi),phi))
        w=np.dot((inv.dot(np.transpose(phi))),t)
    # Construct the weights for non-regularized model
    else:
        w = np.dot(pseudo_inverse,t)
    # Measure root mean squared error on training data.
    predict=np.dot(phi,w)
    square=np.power((t-predict),2)
    train_err=np.sqrt(np.ndarray.mean(square))
    return (w, train_err)


# Define the sigmoid function to help to construct the design-matrix for sigmoid basis
def sigmoid(x,mu,s):
    return 1/(1+np.exp((mu-x)/s))


def design_matrix(basis,x_train,degree,bias,mu,s):
    """ Compute a design matrix Phi from given input datapoints and basis.

    Args:
        basis is string, name of basis to use
        x_train is training inputs
        degree is degree of polynomial to use
        bias is string, indicates if we need bias or not
        mu,s are parameters for sigmoid basis

    Returns:
      phi design matrix
    """

    if basis == 'polynomial':
        if bias=='yes':
            n = np.size(x_train, 0)
            phi = np.full((n, 1), 1)
            for iterator in range(1, degree+1):
                x_train = np.array(x_train)
                x_train_to_degree = np.power(x_train, iterator)
                phi = np.hstack((phi, x_train_to_degree))

        # without bias:
        elif bias=='none':
            phi = x_train
            for iterator in range(2, degree + 1):
                x_train = np.array(x_train)
                x_train_to_degree = np.power(x_train, iterator)
                phi = np.hstack((phi, x_train_to_degree))

    elif basis == 'sigmoid':
        n = np.size(x_train, 0)
        phi = np.full((n, 1), 1)
        for i in mu:
            sigmoid_basis=sigmoid(x_train,i,s)
            phi=np.hstack((phi, sigmoid_basis))

    else:
        assert(False), 'Unknown basis %s' % basis

    return phi


def evaluate_regression(x,t,w,degree,basis,bias,mu,s):
    """Evaluate linear regression on a dataset.

    Args:
      x is the testing input
      t is the testing target
      w is the vector contains weights we calculated from training data
      degree is the degree of polynomial
      basis is string, indicates which basis function we want to use, polynomial or sigmoid
      bias is string, indicates whether we want bias or not
      mu,s are parameter for sigmoid basis
    Returns:
      t_est values of regression on inputs
      err RMS error on training set if t is not None
      """
    phi=design_matrix(basis,x,degree,bias,mu,s)
    t_est = np.dot(phi,w)
    square=np.power((t-t_est),2)
    err = np.sqrt(np.ndarray.mean(square))

    return (t_est, err)
