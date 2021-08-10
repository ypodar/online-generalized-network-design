
import gurobipy as gb
import random
from gurobipy import GRB
import main

abilene = [(1, 0), (1, 3), (0, 3), (0, 2), (2, 5), (3, 4), (4, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 9)]  # 10 nodes
nsf = [(0, 1), (0, 8), (0, 2), (1, 3), (1, 2), (2, 5), (3, 4), (3, 10), (4, 5), (4, 6), (5, 7), (5, 12), (6, 8), (7, 9),
       (8, 9), (9, 13), (9, 11), (10, 13), (11, 12), (12, 13)]  # 14 nodes
# network: list(start, end, weight), outputpairs: list(s,t), alpha: dict(edge:alphaval), qval: dict(edge:qval)
numPairs = 5 # run between 5, 10, and 20
# network, outputpairs, nodes, alpha, qval, sigma = main.networkGen2(numPairs, abilene)

# to run reader - change between nsf and abilene for first 2 parameters. c/d for 3rd, # of pairs to generate for 4th,
# and instance # for 5th
network, outputpairs, nodes, alpha, qval, sigma = main.fileRead(main.fileGen(nsf, "nsf", "d", numPairs, 2))

print("----------Algorithm----------")
output = {}
print("Pairs:", numPairs)
for x in range(10):
    random.shuffle(outputpairs)
    if output.get(main.algo_main(network, outputpairs, alpha, qval, sigma)) is None:
        output[main.algo_main(network, outputpairs, alpha, qval, sigma)] = 1
    else:
        output[main.algo_main(network, outputpairs, alpha, qval, sigma)] += 1

print(output)

print("----------Integer Program----------")
lin_network = gb.tuplelist(network)
linModel = gb.Model("Integer Program")
flow = linModel.addVars(lin_network, vtype=GRB.CONTINUOUS, name="Flow", lb=0)
xval = {}
z = {}
k = len(outputpairs)
tuple_i = gb.tuplelist(list(range(0, k + 1)))  # [0:k]
for i in outputpairs:
    xval[i] = linModel.addVars(lin_network, vtype=GRB.BINARY)
for j in network:
    z[j] = linModel.addVars(tuple_i, vtype=GRB.BINARY)
    linModel.addConstr(flow[j] == gb.quicksum(xval[pair][j] for pair in outputpairs))
for path in xval:
    for node in nodes:
        if node == path[0]:
            val = -1
        elif node == path[1]:
            val = 1
        else:
            val = 0
        linModel.addConstr(gb.quicksum(xval[path][edge] for edge in lin_network.select('*', node, '*')) -
                           gb.quicksum(xval[path][edge] for edge in lin_network.select(node, '*', '*')) == val)

for edge in network:
    linModel.addConstr(gb.quicksum(z[edge][i] for i in tuple_i) == 1)
    for i in range(0, k + 1):
        # linModel.addConstr(k * z[edge][i] >= flow[edge] - i + 1)
        linModel.addConstr(k * (1 - z[edge][i]) >= i - flow[edge])
        linModel.addConstr(k * (1 - z[edge][i]) >= flow[edge] - i)

linModel.setObjective(gb.quicksum(gb.quicksum(((z[edge][i] * (i ** alpha[(edge[0], edge[1])])) for i in tuple_i)) +
                                  (sigma[(edge[0], edge[1])] * (1 - z[edge][0])) for
                                  edge in lin_network), GRB.MINIMIZE)
linModel.setParam("OutputFlag", 0)  # turn off output reporting
linModel.optimize()

print(linModel.objVal)
