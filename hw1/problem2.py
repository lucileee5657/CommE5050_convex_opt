import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


n = 5
p = 3

x = cp.Variable(n)
A = np.array([i+1 for i in range(n)])

constraints = [x >= 0,
               cp.sum(cp.inv_pos(x)) <= 1]


obj = cp.Minimize(cp.pnorm(x, p) + A @ x)
prob = cp.Problem(obj, constraints)
prob.solve()

print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)


objvs = []
ps = [i+1 for i in range(100)]

for p in ps:
    x = cp.Variable(n)
    A = np.array([i+1 for i in range(n)])

    constraints = [x >= 0,
                cp.sum(cp.inv_pos(x)) <= 1]


    obj = cp.Minimize(cp.pnorm(x, p) + A @ x)
    prob = cp.Problem(obj, constraints)
    prob.solve()

    #print("status:", prob.status)
    #print("optimal value", prob.value)
    #print("optimal var", x.value)
    objvs.append(prob.value)
print()

plt.plot(ps, objvs)
plt.xlabel("p")
plt.ylabel("optimal value")
plt.show()