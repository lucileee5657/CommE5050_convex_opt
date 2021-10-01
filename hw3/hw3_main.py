import cvxpy as cp 
import numpy as np 
import time
import matplotlib.pyplot as plt
from hw3_objective import *

def _obj(x, t):
    f1 = (x[0]-1)**2 + (x[0]-1)*x[1] + x[1]**2 - x[2]
    f2 = (x[0]+1)**2 + 1.5*((x[1]-2)**2) - x[2] + 1
    f3 = (x[0]+1)**2 - (x[0]+1)*(x[1]+2) + 1.5*((x[1]+2)**2) - x[2] - 1

    obj = t * x[2] - np.log(-f1) - np.log(-f2) - np.log(-f3)
    return obj

def barrier_method(x, u):
    m = 3.0
    t = 1.0
    alpha = 0.1
    beta = 0.7
    
    epsilon_inner = 1e-5
    epsilon_outer = 1e-8

    # For inner record
    f_l = []
    lambda_l = []
    s_l = []
    
    # For outer record
    outer_iter = 0
    outer_inner = []
    t_list = []
    outer_f = []
    outer_x = []
    
    flag = True
    l = 0
    while (m/t > epsilon_outer):
        while(True):
            o, g, h = my_objective_with_log_barrier(x, t)
            f_l.append(o)

            newton_step = - np.dot(np.linalg.inv(h), g)
            newton_decrement = np.sqrt(-np.dot(np.transpose(g), newton_step))
            lambda_l.append(newton_decrement)
            
            if flag:
                flag = False
                #print("$$$ o: \n", o, "\ng: \n", g, "\nh: \n", h)
                #print("First Newton step: ", newton_step.T, "  First Newton decrement: ", newton_decrement.T)
            
            if ((np.square(newton_decrement)/2) < epsilon_inner):
                outer_f.append(o[0])
                if not outer_inner:
                    outer_inner.append(l)
                else:
                    outer_inner.append(l-sum(outer_inner))
                break

            s = 1.0
            new_x = x + s * newton_step
            while((new_x[0]-1)**2 + (new_x[0]-1)*new_x[1] + new_x[1]**2 - new_x[2] >= 0 or \
                (new_x[0]+1)**2 + 1.5*((new_x[1]-2)**2) - new_x[2] + 1 >= 0 or \
                (new_x[0]+1)**2 - (new_x[0]+1)*(new_x[1]+2) + 1.5*((new_x[1]+2)**2) - new_x[2] - 1 >= 0):
                #print((new_x[0]-1)**2 + (new_x[0]-1)*new_x[1] + new_x[1]**2 - new_x[2])
                #print((new_x[0]+1)**2 + 1.5*((new_x[1]-2)**2) - new_x[2] + 1)
                #print((new_x[0]+1)**2 - (new_x[0]+1)*(new_x[1]+2) + 1.5*((new_x[1]+2)**2) - new_x[2] - 1)
                s = beta * s
                new_x = x + s * newton_step
                #print(s)
                #print(new_x)
                
            new_f =  _obj(new_x, t)
            while(new_f > (o - alpha * s * np.square(newton_decrement))):
                s = s * beta
                new_x = x + s * newton_step
                new_f =  _obj(new_x, t)
            x = x + s * newton_step
            s_l.append(s)
            l += 1
        outer_iter += 1
        t_list.append(t)
        outer_x.append(x)
        t = u * t

    return outer_iter, outer_inner, t_list, outer_f, outer_x, l, x, f_l, lambda_l, s_l

def cvx_method():
    p1 = np.array([[2, 1], [1, 2]])
    p2 = np.array([[2, 0], [0, 3]])
    p3 = np.array([[2, -1], [-1, 3]])

    y1 = np.array([1, 0])
    y2 = np.array([-1, 2])
    y3 = np.array([-1, -2])
    # For n = 2 + 1
    x = cp.Variable(2)
    w = cp.Variable()
    # Objective function
    obj = cp.Minimize(w)
    # Constraints
    constraints = [0.5 * cp.quad_form(x-y1, p1) <= w, 
                    0.5 * cp.quad_form(x-y2, p2)+1 <= w,
                    0.5 * cp.quad_form(x-y3, p3)-1 <= w]
    
    # Solve the problem
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return x.value, w.value, prob.value

def solver():
    init_x = np.array([[0], [0], [10]])
    print("### For u = 20: ")
    u = 20
    outer_iter, outer_inner, t_list, outer_f, outer_x, l, x, f_l, lambda_l, s_l = barrier_method(init_x, u)
    print("Iters for outer loop: ", outer_iter)
    print("Inner loops for each outer loop: ", outer_inner)
    print("t: ", t_list)
    print("f(x) (outer): ", outer_f)
    print("x of each outer: ", outer_x)
    print("Total newton steps: ", l)
    print("\n--------for l---------")
    print("f(x) each step l: ", f_l)
    print("\nlambda: ", lambda_l)
    print("\ns each step l: ", s_l)
    print("-----------------------")
    print("\n***Optimal point of x1 and x2 = ", x[:2].T[0])
    print("***Optimal of w = x3 = ", x[2])
    print("***Optimal value = ", x[2])
    print("======================================================")

    print("### For u = 200: ")
    u = 200
    outer_iter, outer_inner, t_list, outer_f, outer_x, l, x, f_l, lambda_l, s_l = barrier_method(init_x, u)
    print("Iters for outer loop: ", outer_iter)
    print("Inner loops for each outer loop: ", outer_inner)
    print("t: ", t_list)
    print("f(x) (outer): ", outer_f)
    print("x of each outer: ", outer_x)
    print("Total newton steps: ", l)
    print("\n--------for l---------")
    print("f(x) each step l: ", f_l)
    print("\nlambda: ", lambda_l)
    print("\ns each step l: ", s_l)
    print("-----------------------")
    print("\n***Optimal point of x1 and x2 = ", x[:2].T[0])
    print("***Optimal of w = x3 = ", x[2])
    print("***Optimal value = ", x[2])
    print("======================================================")

    print("### For u = 2000: ")
    u = 2000
    outer_iter, outer_inner, t_list, outer_f, outer_x, l, x, f_l, lambda_l, s_l = barrier_method(init_x, u)
    print("Iters for outer loop: ", outer_iter)
    print("Inner loops for each outer loop: ", outer_inner)
    print("t: ", t_list)
    print("f(x) (outer): ", outer_f)
    print("x of each outer: ", outer_x)
    print("Total newton steps: ", l)
    print("\n--------for l---------")
    print("f(x) each step l: ", f_l)
    print("\nlambda: ", lambda_l)
    print("\ns each step l: ", s_l)
    print("-----------------------")
    print("\n***Optimal point of x1 and x2 = ", x[:2].T[0])
    print("***Optimal of w = x3 = ", x[2])
    print("***Optimal value = ", x[2])
    print("======================================================")

    print("### For u = 20000: ")
    u = 20000
    outer_iter, outer_inner, t_list, outer_f, outer_x, l, x, f_l, lambda_l, s_l = barrier_method(init_x, u)
    print("Iters for outer loop: ", outer_iter)
    print("Inner loops for each outer loop: ", outer_inner)
    print("t: ", t_list)
    print("f(x) (outer): ", outer_f)
    print("x of each outer: ", outer_x)
    print("Total newton steps: ", l)
    print("\n--------for l---------")
    print("f(x) each step l: ", f_l)
    print("\nlambda: ", lambda_l)
    print("\ns each step l: ", s_l)
    print("-----------------------")
    print("\n***Optimal point of x1 and x2 = ", x[:2].T[0])
    print("***Optimal of w = x3 = ", x[2])
    print("***Optimal value = ", x[2])
    print("======================================================")

    print("### Using cvx tool...")
    opt_x, opt_w, opt_value = cvx_method()
    print("***Optimal point of x1 and x2 = ", opt_x)
    print("***Optimal of w = x3 = ", opt_w)
    print("***Optimal value = ", opt_value)

solver()