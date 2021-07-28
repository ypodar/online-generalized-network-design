import cvxpy as cp
import random
import main
from math import e
import gurobipy as gb
from gurobipy import GRB

# (initial, end, weight), (s,t), ((initial, end) : alpha)
network, outputpairs, alpha, qval, sigma = main.algo_main()
nodes = []
for edge in network:
    if (edge[0] == '0') and (edge[1] not in nodes):
        nodes.append(edge[1])
    elif (edge[1] == '0') and (edge[0] not in nodes):
        nodes.append(edge[0])


# make a dictionary for flow where key = (initial, end , weight) from network
# nested dictionary for x(i,e) values. main key = (s, t) from output pairs
# value - dictionary with edge from network
# values for each is cp.Variable()
flow = {}
xval = {}
constraints = []
k = len(outputpairs)
linModel = gb.Model("Linear Model")
alpha_max = max(alpha.values())
flow_lin = {}
xval_lin = {}
z = {}
lin_obj = []
for j in outputpairs:
    xval[j] = {}
    xval_lin[j] = {}
    for i in network:
        xval[j][i] = cp.Variable(nonneg=True)

        xval_lin[j][i] = linModel.addVar()
        linModel.addConstr(xval_lin[j][i] >= 0)
for i in network:
    flow[i] = cp.Variable(nonneg=True)

    flow_lin[i] = linModel.addVar()
    linModel.addConstr(flow_lin[i] >= 0)
    z[i] = {}
    for l in range(1, k + 1):
        z[i][l] = linModel.addVar()
        lin_obj.append((l ** alpha_max) * z[i][l])
        linModel.addConstr((k * z[i][l]) - (flow[i] - l + 1) >= 0)
        linModel.addConstr((k * (1 - z[i][l])) - l + flow[i] >= 0)


objsum = 0
objsum2 = 0
for i in flow:
    temp = (i[0], i[1])
    objsum += i[2] * (flow[i] ** alpha[temp])
    objsum2 += (flow[i] ** alpha[temp]) + (alpha[temp] * flow[i] / (e ** alpha[temp])) + \
               (flow[i] * (qval[temp] ** (alpha[temp] - 1)))
obj1 = cp.Minimize(objsum)
obj2 = cp.Minimize(objsum2)


# constraint - each f(e) = sum of x(i) for that given edge
for i in network:
    constraint_sum = 0
    for j in outputpairs:
        constraint_sum += xval[j][i]
    constraints.append(flow[i] == constraint_sum)

# difference of edges entering and leaving the node based on if node is start end or neither
# {key:{key:value}} #1 key - (s,t) pair, #2 key - (intial, end, weight)
for path in xval:
    edges = xval[path]
    for node in nodes:
        in_sum = 0
        out_sum = 0
        for edge in edges:
            if edge[1] == node:
                in_sum += edges[edge]
            elif edge[0] == node:
                out_sum += edges[edge]
        if node == path[0]:
            constraints.append(in_sum - out_sum == -1)

        elif node == path[1]:
            constraints.append(in_sum - out_sum == 1)
        else:
            constraints.append(in_sum - out_sum == 0)


problem = cp.Problem(obj1, constraints)
problem1 = cp.Problem(obj2, constraints)

print("Convex Program Value 1:", problem.solve())
x = problem1.solve()
print("Convex Program Value 2:", x)
print("relaxed value: ", (x / (1 + (2 * (e ** -2)))))


linModel.setObjective(sum(lin_obj))
linModel.optimize()
print("Integer Model:", linModel.objVal)