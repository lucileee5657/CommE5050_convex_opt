import numpy as np
import cvxpy as cp 

# minimize 2x + y
# s.t. x >= 0, y >= 0, 1/x + 1/y <= 1


x = cp.Variable()
y = cp.Variable()

constraints = [x >= 0,
               y >= 0,
               cp.inv_pos(x) + cp.inv_pos(y) <= 1]

obj = cp.Minimize(2*x + y)
prob = cp.Problem(obj, constraints)
prob.solve()

print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)