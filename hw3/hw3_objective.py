import cvxpy as cp 
import numpy as np 
import time

def my_objective_with_log_barrier(x, t):
    
    f1 = (x[0]-1)**2 + (x[0]-1)*x[1] + x[1]**2 - x[2]
    f2 = (x[0]+1)**2 + 1.5*((x[1]-2)**2) - x[2] + 1
    f3 = (x[0]+1)**2 - (x[0]+1)*(x[1]+2) + 1.5*((x[1]+2)**2) - x[2] - 1

    obj = t * x[2] - np.log(-f1) - np.log(-f2) - np.log(-f3)

    g_f1 = np.array([[2 * (x[0]-1) + x[1]],[(x[0]-1) + 2 * x[1]],[-1]], dtype=object)
    g_f2 = np.array([[2 * (x[0]+1)],[3 * (x[1]-2)],[-1]], dtype=object)
    g_f3 = np.array([[2 * (x[0]+1) - (x[1] + 2)],[-(x[0]+1) + 3 * (x[1]+2)],[-1]], dtype=object)
    g_f0 = np.array([[0], [0], [t]])
    gradient = g_f0 - (1/f1) * g_f1 - (1/f2) * g_f2 - (1/f3) * g_f3

    h_f1 = np.array([[2, 1, 0],[1, 2, 0],[0, 0, 0]])
    h_f2 = np.array([[2, 0, 0],[0, 3, 0],[0, 0, 0]])
    h_f3 = np.array([[2, -1, 0],[-1, 3, 0],[0, 0, 0]])

    hessian = (1/f1**2) * np.dot(g_f1, g_f1.T) \
            + (1/f2**2) * np.dot(g_f2, g_f2.T) \
            + (1/f3**2) * np.dot(g_f3, g_f3.T) \
            - (1/f1) * h_f1 \
            - (1/f2) * h_f2 \
            - (1/f3) * h_f3

    for i in range(len(gradient)-1):
        gradient[i] = gradient[i][0]
    for i in range(len(hessian)):
        for j in range(len(hessian[i])):
            if i == 2 and j == 2:
                pass
            else:
                hessian[i][j] = hessian[i][j][0]
    gradient = np.array(gradient, dtype='float')
    hessian = np.array(hessian, dtype='float')
    return obj, gradient.reshape((3, 1)), hessian.reshape((3, 3))

