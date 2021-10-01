import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# c = 0.5, 1, 2 ==> l = -1, u = 1.2
# c = 3, 4 ==> l = -1, u = 1.1
c = 2
err = 1e-8
l = -1
u = 1.1
p = None
v = None

print("(c) " + "-"*40)
print("Using qcp solver...")
x = cp.Variable(2, pos=True)
constraints = [x[0] + c * x[1] <= 1,
                cp.norm(x) <= 1 / (2**0.5)]
obj = cp.Maximize((x[0] * x[1]))

prob = cp.Problem(obj, constraints)
prob.solve(qcp=True)
print("optimal value p* ", prob.value)
print("optimal point x* ", x.value)


print("Bisection method...")
while (u - l) >= err:
    t = (l + u) / 2

    x = cp.Variable(2, pos=True)
    constraints = [x[0] + c * x[1] <= 1,
                cp.norm(x) <= 1 / (2**0.5),
                cp.power(cp.inv_pos(cp.geo_mean(x)), 2) <= 1/t] 
                # equal to cp.inv_prod(x) <= 1/t
                # but there 's error when I use cp.inv_prod(x)
    
    obj = cp.Maximize(0)
    
    prob = cp.Problem(obj, constraints)
    prob.solve()

    #print("l: ", l, "t: ", t, "u: ", u)
    #print("status:", prob.status)
    #print("optimal value p* ", prob.value)
    #print("optimal point x* ", x.value)
    if prob.status == "infeasible":
        u = t
    else:
        l = t
        p = prob
        v = x 
if v:
    print("-"*50)
    print("value of t: ", t) 
    print("optimal value p* ", v.value[0]*v.value[1])
    print("optimal point x* ", v.value)


#######################################################
print("\n\n(d) " + "-"*40)
c_list = [0.5, 1, 2, 3, 4]
opt_list = [0]*5
for i, c in enumerate(c_list):
    print("When c = ", c)
    err = 1e-8
    l = -1
    u = 1.1
    while (u - l) >= err:
        t = (l + u) / 2

        x = cp.Variable(2, pos=True)
        constraints = [x[0] + c * x[1] <= 1,
                    cp.norm(x) <= 1 / (2**0.5),
                    cp.power(cp.inv_pos(cp.geo_mean(x)), 2) <= 1/t] 
                    # equal to cp.inv_prod(x) <= 1/t
                    # but there 's error when I use cp.inv_prod(x)
        
        obj = cp.Maximize(0)
        
        prob = cp.Problem(obj, constraints)
        prob.solve()

        if prob.status == "infeasible":
            u = t
        else:
            l = t
            p = prob
            v = x 
    if v:
        print("value of t: ", t) 
        print("optimal value p* ", v.value[0]*v.value[1])
        print("optimal point x* ", v.value)
        opt_list[i] = v.value[0]*v.value[1]

plt.plot(c_list, opt_list)
plt.xlabel("c")
plt.ylabel("optimal value p*")
plt.show()