import cvxpy as cp 
import numpy as np 
import time
import matplotlib.pyplot as plt

def my_objective_(x, A, b, c):
    obj = np.dot(np.transpose(c), x) - \
        np.log(-np.dot(A[0], x) + b[0]) - \
        np.log(-np.dot(A[1], x) + b[1]) - \
        np.log(-np.dot(A[2], x) + b[2])
    gradient = np.transpose(np.transpose(c) + \
        (1/(-np.dot(A[0], x) + b[0])) * (A[0]) +\
        (1/(-np.dot(A[1], x) + b[1])) * (A[1]) +\
        (1/(-np.dot(A[2], x) + b[2])) * (A[2]))
    hessian = (1/(-np.dot(A[0], x) + b[0])**2) * np.dot(np.transpose([A[0]]), [A[0]]) +\
        (1/(-np.dot(A[1], x) + b[1])**2) * np.dot(np.transpose([A[1]]), [A[1]]) +\
        (1/(-np.dot(A[2], x) + b[2])**2) * np.dot(np.transpose([A[2]]), [A[2]])
    return obj, gradient, hessian