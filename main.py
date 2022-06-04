from collections import defaultdict
import random
from math import e
import numpy
import ast
import gurobipy as gb
from gurobipy import GRB


class adjList:
    def __init__(self):
        self.edges = defaultdict(list)
        self.weights = {}
        self.load = {}
        self.psi = {}

    # adding edge with weight for undirected weighted graph

    # def add_edge(self, from_node, to_node, weight):
    #     # Note: assumes edges are uni-directional
    #     self.edges[from_node].append(to_node)
    #     self.edges[to_node].append(from_node)
    #     self.weights[(from_node, to_node)] = weight
    #     self.weights[(to_node, from_node)] = weight
    #     self.load[(from_node, to_node)] = 0
    #     self.load[(to_node, from_node)] = 0

    # creates a bidirected edge from input of one i,j pair
    def add_edge(self, from_node, to_node):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.load[(from_node, to_node)] = 0
        self.load[(to_node, from_node)] = 0


# finds shortest path between initial and end node, default is using graph.psi, else uses graph.weights
# returns path in the form of [initial, ... , end]
def dijkstra(graph, initial, end, psi=True):
    # shortest_paths is the dictionary that stores current best paths
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        neighbors = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in neighbors:
            if psi:
                weight = graph.psi[(current_node, next_node)] + weight_to_current_node
            else:
                weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            print("Route Not Possible")
            return False
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path


def network_generator(k, edges, nodes, vtype="c"):
    alpha = {}
    sigma = {}
    outputpairs = []
    for i in edges:
        temp = (i[1], i[0])
        if vtype.lower() == "d":
            # discrete
            alpha[i] = random.randint(2, 3)
        else:
            # continuous
            alpha[i] = random.uniform(1.1, 3)
        sigma[i] = random.uniform(1, (0.3 * k) ** alpha[i])
        alpha[temp] = alpha[i]
        sigma[temp] = sigma[i]
    j = 0
    while j < k:
        s = random.choice(nodes)
        t = random.choice(nodes)
        if s != t and (s, t) not in outputpairs:
            outputpairs.append((s, t))
            j += 1
    return outputpairs, nodes, alpha, sigma


# graph is an adjacency list (dict - key:node, value: list)
# returns true if a path from start to end exists
def bfs(graph, start, end):
    if start == end:
        return True
    visited = {}
    for node in graph:
        visited[node] = False
    visited = [start]
    queue = [start]
    while queue:
        node = queue[0]
        queue.pop(0)
        for neighbor in graph[node]:
            if neighbor == end:
                return True
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
    return False


# checks if every node is accessible by a single node
# and therefore every node is accessible by every other node
def graph_connected(nodes, edges):
    adj_matrix = defaultdict(list)
    for i in edges:
        adj_matrix[i[0]].append(i[1])
        adj_matrix[i[1]].append(i[0])
    # source is an arbitrary node
    source = random.choice(nodes)
    for dest in nodes:
        if not bfs(adj_matrix, source, dest):
            return False
    return True


def generateEdges(numNodes):
    numEdges = int(numNodes * numpy.log(numNodes)) + 1
    nodes = []
    for i in range(numNodes):
        nodes.append(i)
    j = 0
    edges = []
    while j < numEdges:
        a = random.choice(nodes)
        b = random.choice(nodes)
        if a != b and (a, b) not in edges and (b, a) not in edges:
            edges.append((a, b))
            j += 1
    if graph_connected(nodes, edges):
        return nodes, edges
    else:
        return generateEdges(numNodes)


# vtype:continuous(c) or discrete(d) alpha values, k: num of pairs, num: instance #
# alpha, sigma, and pairs to be written to file
def fileGen(edges, nodes, networkname, vtype, k, num):
    # alpha and sigma values are bidirected after network_generator
    outputpairs, nodes, alpha, sigma = network_generator(k, edges, nodes, vtype)
    string = networkname + "_n" + str(len(nodes)) + "_" + vtype + "_k" + str(k) + "_" + str(num) + ".txt"
    f = open(string, "w+")
    f.write(str(alpha) + "\n")
    f.write(str(sigma) + "\n")
    f.write(str(outputpairs) + "\n")
    f.write(str(nodes) + "\n")
    f.close()
    print(string)
    return string


# reads from a file, computes qval and network
def fileRead(filename):
    f = open(filename, "r")
    alpha = ast.literal_eval(f.readline())
    sigma = ast.literal_eval(f.readline())
    pairs = ast.literal_eval(f.readline())
    nodes = ast.literal_eval(f.readline())
    network = []
    qval = {}
    for edge in alpha:
        # network now stores all the bidirected edges
        network.append((edge[0], edge[1]))
        qval[edge] = sigma[edge] ** (1 / alpha[edge])
    return network, pairs, nodes, alpha, qval, sigma


# calculates psi values for given load and weights for graph, alpha can be adjusted (default 2)
# sigma has to be added (and then formula changed)
def psicalc(graph, alpha, qval):
    for f_node in graph.edges:
        for t_node in graph.edges[f_node]:
            graph.psi[(f_node, t_node)] = \
                qval[(f_node, t_node)] ** (alpha[(f_node, t_node)] - 1) \
                + alpha[(f_node, t_node)] * (graph.load[(f_node, t_node)] / (e * alpha[(f_node, t_node)])) ** (
                        alpha[(f_node, t_node)] - 1) \
                + alpha[(f_node, t_node)] / (e ** alpha[(f_node, t_node)])


# takes in a pair, performs psi calculation, finds shortest path for pair and then updates graph.load
# returns path taken for storing
def algorithm(graph, pair, alpha, qval):
    psicalc(graph, alpha, qval)
    path = dijkstra(graph, pair[0], pair[1])
    for i in range(len(path) - 1):
        graph.load[(path[i], path[i + 1])] += 1
        graph.load[(path[i + 1], path[i])] += 1
    return path


# generates a network and pairs, continues to run until all pairs are completed
def algo_main(network, pairs, alpha, qval, sigma):
    x = adjList()
    undirected_network = []
    for i in range(len(network)):
        a = network[i]
        if network[i] not in undirected_network and (a[1], a[0]) not in undirected_network:
            undirected_network.append(network[i])
    algo_sum = 0
    for edge in undirected_network:
        x.add_edge(*edge)
    for i in pairs:
        algorithm(x, i, alpha, qval)
    for j in undirected_network:
        #  algo_sum += x.weights[j] * (x.load[j] ** alpha[j])
        algo_sum += x.load[j] ** alpha[j]
        if x.load[j] >= 1:
            algo_sum += sigma[j]
    return algo_sum

rerun_list = [
    "runs/abilene_c_k5_1.txt",
    "runs/abilene_c_k5_2.txt",
    "runs/abilene_c_k10_1.txt",
    "runs/abilene_c_k10_2.txt",
    "runs/abilene_c_k20_1.txt",
    "runs/abilene_c_k20_2.txt",
    "runs/abilene_d_k5_1.txt",
    "runs/abilene_d_k5_2.txt",
    "runs/abilene_d_k10_1.txt",
    "runs/abilene_d_k10_2.txt",
    "runs/abilene_d_k20_1.txt",
    "runs/abilene_d_k20_2.txt",
    "runs/nsf_c_k5_1.txt",
    "runs/nsf_c_k5_2.txt",
    "runs/nsf_c_k10_1.txt",
    "runs/nsf_c_k10_2.txt",
    "runs/nsf_c_k20_1.txt",
    "runs/nsf_c_k20_2.txt",
    "runs/nsf_d_k5_1.txt",
    "runs/nsf_d_k5_2.txt",
    "runs/nsf_d_k10_1.txt",
    "runs/nsf_d_k10_2.txt",
    "runs/nsf_d_k20_1.txt",
    "runs/nsf_d_k20_2.txt"
]


def algorithm_run(network, outputpairs, alpha, qval, sigma):
    print("----------Algorithm----------")
    for x in range(10):
        random.shuffle(outputpairs)
        print(algo_main(network, outputpairs, alpha, qval, sigma))



def IPrun(network, outputpairs, alpha, qval, sigma):
    print("----------Integer Program----------")
    undirected_network = []
    for i in range(len(network)):
        a = network[i]
        if network[i] not in undirected_network and (a[1], a[0]) not in undirected_network:
            undirected_network.append(network[i])
    lin_network = gb.tuplelist(network)
    lin_undirected_network = gb.tuplelist(undirected_network)
    linModel = gb.Model("Integer Program")
    flow = linModel.addVars(lin_network, vtype=GRB.CONTINUOUS, name="Flow", lb=0)
    xval = {}
    z = {}
    k = len(outputpairs)
    tuple_i = gb.tuplelist(list(range(0, k + 1)))  # [0:k]
    for i in outputpairs:
        xval[i] = linModel.addVars(lin_network, vtype=GRB.BINARY)
    for j in undirected_network:
        z[j] = linModel.addVars(tuple_i, vtype=GRB.BINARY)
        reverse_arc = (j[1], j[0])
        linModel.addConstr(flow[j] == gb.quicksum(xval[pair][j] + xval[pair][reverse_arc] for pair in outputpairs))
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

    for edge in undirected_network:
        reverse_arc = (edge[1], edge[0])
        linModel.addConstr(gb.quicksum(z[edge][i] for i in tuple_i) == 1)
        for i in range(0, k + 1):
            # linModel.addConstr(k * z[edge][i] >= flow[edge] - i + 1)
            linModel.addConstr(k * (1 - z[edge][i]) >= i - (flow[edge] + flow[reverse_arc]))
            linModel.addConstr(k * (1 - z[edge][i]) >= (flow[edge] + flow[reverse_arc]) - i)

    # calculate flow of edge i, j and j, i (but don't double count - only count for 1 edge)
    # could use only when i < j
    linModel.setObjective(gb.quicksum(gb.quicksum(((z[edge][i] * (i ** alpha[(edge[0], edge[1])])) for i in tuple_i)) +
                                      (sigma[(edge[0], edge[1])] * (1 - z[edge][0])) for
                                      edge in lin_undirected_network), GRB.MINIMIZE)
    linModel.setParam("OutputFlag", 0)  # turn off output reporting
    # time limit is only for computation, model building time is not included
    linModel.setParam('TimeLimit', 500)
    linModel.optimize()

    print("Objective Value:")
    print(linModel.objVal)
    print("Best Lower Bound:")
    print(linModel.ObjBound)
    print("Runtime:")
    print(linModel.Runtime)

rerun_list = [
    "runs/abilene_c_k5_1.txt",
    "runs/abilene_c_k5_2.txt",
    "runs/abilene_c_k10_1.txt",
    "runs/abilene_c_k10_2.txt",
    "runs/abilene_c_k20_1.txt",
    "runs/abilene_c_k20_2.txt",
    "runs/abilene_d_k5_1.txt",
    "runs/abilene_d_k5_2.txt",
    "runs/abilene_d_k10_1.txt",
    "runs/abilene_d_k10_2.txt",
    "runs/abilene_d_k20_1.txt",
    "runs/abilene_d_k20_2.txt",
    "runs/nsf_c_k5_1.txt",
    "runs/nsf_c_k5_2.txt",
    "runs/nsf_c_k10_1.txt",
    "runs/nsf_c_k10_2.txt",
    "runs/nsf_c_k20_1.txt",
    "runs/nsf_c_k20_2.txt",
    "runs/nsf_d_k5_1.txt",
    "runs/nsf_d_k5_2.txt",
    "runs/nsf_d_k10_1.txt",
    "runs/nsf_d_k10_2.txt",
    "runs/nsf_d_k20_1.txt",
    "runs/nsf_d_k20_2.txt"
]
#
# for file in rerun_list:
#     print(file)
#     network, outputpairs, nodes, alpha, qval, sigma = fileRead(file)
#     algorithm_run(network, outputpairs, alpha, qval, sigma)
#     IPrun(network, outputpairs, alpha, qval, sigma)
#     print("--------------------------------")

abilene = [(1, 0), (1, 3), (0, 3), (0, 2), (2, 5), (3, 4), (4, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 9)]  # 10 nodes
nsf = [(0, 1), (0, 8), (0, 2), (1, 3), (1, 2), (2, 5), (3, 4), (3, 10), (4, 5), (4, 6), (5, 7), (5, 12), (6, 8), (7, 9),
       (8, 9), (9, 13), (9, 11), (10, 13), (11, 12), (12, 13)]  # 14 nodes
# network: list(start, end, weight), outputpairs: list(s,t), alpha: dict(edge:alphaval), qval: dict(edge:qval)

# to run reader - change between nsf and abilene for first 2 parameters. c/d for 3rd, # of pairs to generate for 4th,
# and instance # for 5th
numPairs = 40  # run between 5, 10, and 20
nodes, edges = generateEdges(100)

# fileGen inputs: list of edges and nodes, networkname, c/d (cont or discrete), number of pairs, and instance #
# network, outputpairs, nodes, alpha, qval, sigma = fileRead(fileGen(edges, nodes, "random", "d", numPairs, 2))
# network, outputpairs, nodes, alpha, qval, sigma = fileRead("random_n50_c_k20_1.txt")
# algorithm_run(network, outputpairs, nodes, alpha, qval, sigma)
# IPrun(network, outputpairs, nodes, alpha, qval, sigma)
