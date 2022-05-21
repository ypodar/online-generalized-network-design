
import gurobipy as gb
import random
from gurobipy import GRB
import main

abilene = [(1, 0), (1, 3), (0, 3), (0, 2), (2, 5), (3, 4), (4, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 9)]  # 10 nodes
nsf = [(0, 1), (0, 8), (0, 2), (1, 3), (1, 2), (2, 5), (3, 4), (3, 10), (4, 5), (4, 6), (5, 7), (5, 12), (6, 8), (7, 9),
       (8, 9), (9, 13), (9, 11), (10, 13), (11, 12), (12, 13)]  # 14 nodes
# network: list(start, end, weight), outputpairs: list(s,t), alpha: dict(edge:alphaval), qval: dict(edge:qval)

# to run reader - change between nsf and abilene for first 2 parameters. c/d for 3rd, # of pairs to generate for 4th,
# and instance # for 5th
numPairs = 20 # run between 5, 10, and 20
temp_nodes, temp_edges = main.generateEdges(50)
network, outputpairs, nodes, alpha, qval, sigma = main.fileRead(main.fileGen(temp_edges, temp_nodes, "test1", "c", numPairs, 1))
#network, outputpairs, nodes, alpha, qval, sigma = main.fileRead("nsf_d_k20_1.txt")
# network, outputpairs, nodes, alpha, qval, sigma = main.fileRead(".txt")
# network, outputpairs, nodes, alpha, qval, sigma = main.networkGen2(numPairs, abilene)

print("----------Algorithm----------")
print("Pairs:", numPairs)
for x in range(10):
    random.shuffle(outputpairs)
    # if output.get(main.algo_main(network, outputpairs, alpha, qval, sigma)) is None:
    #     output[main.algo_main(network, outputpairs, alpha, qval, sigma)] = 1
    # else:
    #     output[main.algo_main(network, outputpairs, alpha, qval, sigma)] += 1
    print(main.algo_main(network, outputpairs, alpha, qval, sigma))


print("----------Integer Program----------")
undirected_network = []
for i in range(len(network)):
    a = network[i]
    if network[i] not in undirected_network and (a[1], a[0]) not in undirected_network:
        undirected_network.append(network[i])
lin_directed_network = gb.tuplelist(network)
lin_undirected_network = gb.tuplelist(undirected_network)
# create undirected network (remove the j, i edges for each i, j)
linModel = gb.Model("Integer Program")
flow = linModel.addVars(lin_directed_network, vtype=GRB.CONTINUOUS, name="Flow", lb=0)
xval = {}
z = {}
k = len(outputpairs)
tuple_i = gb.tuplelist(list(range(0, k + 1)))  # [0:k]
for i in outputpairs:
    xval[i] = linModel.addVars(lin_directed_network, vtype=GRB.BINARY)
for j in undirected_network:
    z[j] = linModel.addVars(tuple_i, vtype=GRB.BINARY)
for e in network:
    linModel.addConstr(flow[e] == gb.quicksum(xval[pair][e] for pair in outputpairs))
for path in xval:
    for node in nodes:
        if node == path[0]:
            val = -1
        elif node == path[1]:
            val = 1
        else:
            val = 0
        linModel.addConstr(gb.quicksum(xval[path][edge] for edge in lin_directed_network.select('*', node)) -
                           gb.quicksum(xval[path][edge] for edge in lin_directed_network.select(node, '*')) == val)

for edge in undirected_network:
    reverse_arc = (edge[1], edge[0])
    linModel.addConstr(gb.quicksum(z[edge][i] for i in tuple_i) == 1)
    for i in range(0, k + 1):
        # linModel.addConstr(k * z[edge][i] >= flow[edge] - i + 1)
        linModel.addConstr(k * (1 - z[edge][i]) >= i - (flow[edge] + flow[reverse_arc]))
        linModel.addConstr(k * (1 - z[edge][i]) >= (flow[edge] + flow[reverse_arc]) - i)

# calculate flow of edge i, j and j, i (but don't double count - only count for 1 edge)
# could use only when i < j
linModel.setObjective(gb.quicksum(gb.quicksum(((z[edge][i] * (i ** alpha[edge])) for i in tuple_i)) +
                                  (sigma[edge] * (1 - z[edge][0])) for
                                  edge in lin_undirected_network), GRB.MINIMIZE)
# linModel.setParam("OutputFlag", 0)  # turn off output reporting
linModel.optimize()

# set a time limit of 500s

print(linModel.objVal)
# print("Network")
# print(network)
# print("Xval")
# print(xval)
# print("Z")
# print(z)
