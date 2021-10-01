import cvxpy as cp 
import numpy as np 
import time
import matplotlib.pyplot as plt
from hw2_objective import *

# For 1.(c)
'''
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
'''

def newton_method(x, A, b, c):
    t = 1.0
    alpha = 0.1
    beta = 0.7
    epsilon = 1e-10
    
    # For 1.(j)
    obj_k = []
    decrement_k = []
    t_k = []
    t_k.append(t)
    flag = True
    while(True):
        
        o, g, h = my_objective_(x, A, b, c)
        obj_k.append(o[0][0])

        # Compute the Newton step and decrement
        # For 1.(e)
        newton_step = - np.dot(np.linalg.inv(h), g)
        # For 1.(f)
        newton_decrement = np.sqrt(-np.dot(np.transpose(g), newton_step))
        decrement_k.append(newton_decrement[0][0])
        if flag:
            flag = False
            print("First Newton step: ", newton_step.T, "  First Newton decrement: ", newton_decrement.T)
        

        # Stopping criterion
        # For 1.(i)
        if ((np.square(newton_decrement)/2) < epsilon):
            return x, obj_k, decrement_k, t_k

        # Line search
        # For 1.(g)
        new_x = x + t * newton_step

        # For 1.(h)
        new_f =  np.dot(np.transpose(c), new_x) - \
                np.log(-np.dot(A[0], new_x) + b[0]) - \
                np.log(-np.dot(A[1], new_x) + b[1]) - \
                np.log(-np.dot(A[2], new_x) + b[2])
        while(new_f > (o - alpha * t * np.square(newton_decrement))):
            t = t * beta
            new_x = x + t * newton_step
            new_f =  np.dot(np.transpose(c), new_x) - \
                np.log(-np.dot(A[0], new_x) + b[0]) - \
                np.log(-np.dot(A[1], new_x) + b[1]) - \
                np.log(-np.dot(A[2], new_x) + b[2])
        
        # Update
        x = x + t * newton_step
        t_k.append(t)

def cvx_method(A, b, c):
    # For n = 2
    x = cp.Variable(2)
    # Objective function
    obj = cp.Minimize(c.T @ x - \
                    cp.log(-A[0] @ x + b[0]) - \
                    cp.log(-A[1] @ x + b[1]) - \
                    cp.log(-A[2] @ x + b[2]))
    # Solve the problem
    prob = cp.Problem(obj)
    prob.solve()
    return x.value, prob.value


def solver():
    # For 1(d)
    print("For original A, b, c: ")
    A = np.array([[1, 3],[1, -3],[-1, 0]])
    b = np.array([[0.1], [0.1], [0.1]])
    c = np.array([[1], [2]])
    x_init = np.array([[0], [0]])
    print("A: \n", A)
    print("b: \n", b)
    print("c: \n", c)
    print("Getting result from the Newton's method...")
    x, obj_k, decrement_k, t_k = newton_method(x_init, A, b, c)
    print("***Optimal point of x = ", x.T)
    print("***Optimal value = ", obj_k[-1])
    print("Obj value in each iter: \n", obj_k)
    print("Decrement in each iter: \n", decrement_k)
    print("t in each iter: \n", t_k)
    decrement_k2 = [i**2 / 2 for i in decrement_k]
    plt.plot(decrement_k2)
    plt.xlabel("k")
    plt.ylabel("(lambda(x^(k))**2 / 2)")
    plt.show()

    print("Getting result from cvx toolbox...")
    x, p = cvx_method(A, b, c)
    print("***Optimal point of x = ", x)
    print("***Optimal value = ", p)
    print("\n===================================================\n")
    
    print("For A, b, c in 1.(l): ")
    A = np.array([[1, 3],[2, -3],[-1, 0]])
    b = np.array([[0.2], [0.3], [0.4]])
    c = np.array([[1], [1]])
    print("A: \n", A)
    print("b: \n", b)
    print("c: \n", c)
    print("Getting result from the Newton's method...")
    x, obj_k, decrement_k, t_k = newton_method(x_init, A, b, c)
    print("***Optimal point of x = ", x.T)
    print("***Optimal value = ", obj_k[-1])
    print("Obj value in each iter: \n", obj_k)
    print("Decrement in each iter: \n", decrement_k)
    print("t in each iter: \n", t_k)
    decrement_k2 = [i**2 / 2 for i in decrement_k]
    plt.plot(decrement_k2)
    plt.xlabel("k")
    plt.ylabel("(lambda(x^(k))**2 / 2)")
    plt.show()

    print("Getting result from cvx toolbox...")
    x, p = cvx_method(A, b, c)
    print("***Optimal point of x = ", x)
    print("***Optimal value = ", p)

solver()


    



