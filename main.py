# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from collections import defaultdict
import random
from math import e
import ast



class aList:
    def __init__(self):
        self.edges = defaultdict(list)
        self.weights = {}
        self.load = {}
        self.psi = {}

    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are uni-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight
        self.load[(from_node, to_node)] = 0
        self.load[(to_node, from_node)] = 0
    def add_edge(self, from_node, to_node):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.load[(from_node, to_node)] = 0
        self.load[(to_node, from_node)] = 0


# finds shortest path between initial and end node, default is using graph.psi, else uses graph.weights
# returns path in the form of [initial, ... , end]
def dijkstra2(graph, initial, end, psi=True):
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
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
            return "Route Not Possible"
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


# string = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnoqrstuvwxyz"
max_weight = 15
max_nodes = 50
num_edges = 150


# generates a strong network, returns a list with tuples in the form (initial, end, weight)
# network is a list with tuples in the form (initial, end, weight)
# pairs is  a list of (x,y) pairs that exist in the network (does not include pairs to the central node)
# edges is the list of all nodes present in the network
# outputpairs is a list of (s,t) pairs to test algorithm against. Randomized to value of k
# k is the number of pairs that has to be outputted
# ('A', 'B', 6)
def networkGen(k):
    network = []
    pairs = []
    nodes = []
    alpha = {}
    sigma = {}
    qval = {}
    outputpairs = []
    i = 0
    j = 0
    while i < num_edges:
        weight = random.randint(1, max_weight)
        x = random.choice(list(range(max_nodes)))
        y = random.choice(list(range(max_nodes)))
        if x != y and (x, y) not in pairs:
            pairs.append((x, y))
            network.append((x, y, weight))
            alpha[(x, y)] = random.randint(2, 3)
            sigma[(x, y)] = random.uniform(1, (0.3 * k) ** alpha[(x, y)])
            qval[(x, y)] = sigma[(x, y)] ** (1 / alpha[(x, y)])
            if x not in nodes:
                nodes.append(x)
                w1 = random.randint(max_weight * 2, max_weight * 3)
                w2 = random.randint(max_weight * 2, max_weight * 3)
                network.append((x, '0', w1))
                network.append(('0', x, w2))
                alpha[(x, '0')] = random.randint(2, 3)
                alpha[('0', x)] = random.randint(2, 3)
                sigma[(x, '0')] = random.uniform(1, (0.3 * k) ** alpha[(x, '0')])
                sigma[('0', x)] = random.uniform(1, (0.3 * k) ** alpha[('0', x)])
                qval[(x, '0')] = sigma[(x, '0')] ** (1 / alpha[(x, '0')])
                qval[('0', x)] = sigma[('0', x)] ** (1 / alpha[('0', x)])
            if y not in nodes:
                nodes.append(y)
                w1 = random.randint(max_weight * 2, max_weight * 3)
                w2 = random.randint(max_weight * 2, max_weight * 3)
                network.append((y, '0', w1))
                network.append(('0', y, w2))
                alpha[(y, '0')] = random.randint(2, 3)
                alpha[('0', y)] = random.randint(2, 3)
                sigma[(y, '0')] = random.uniform(1, (0.3 * k) ** alpha[(y, '0')])
                sigma[('0', y)] = random.uniform(1, (0.3 * k) ** alpha[('0', y)])
                qval[(y, '0')] = sigma[(y, '0')] ** (1 / alpha[(y, '0')])
                qval[('0', y)] = sigma[('0', y)] ** (1 / alpha[('0', y)])
            i += 1
    while j < k:
        s = random.choice(nodes)
        t = random.choice(nodes)
        if s != t and (s, t) not in outputpairs:
            outputpairs.append((s, t))
            j += 1
    print("Nodes:", max_nodes)
    nodes.append('0')
    return network, outputpairs, nodes, alpha, qval, sigma


# takes in edges as a list of (start, node) points. Generates network with each edge and weight, alpha, qval, sigma and
# k pairs. vtype is 'c' for continuous alpha values [1.1, 3], 'd' for discrete alpha values = [2,3]
# defaults to continuous alpha values
# output pairs is list of nodes to run data to/from
def networkGen2(k, edges, vtype="c"):
    network = []
    nodes = []
    alpha = {}
    sigma = {}
    qval = {}
    outputpairs = []
    for i in edges:
        if i[0] not in nodes:
            nodes.append(i[0])
        if i[1] not in nodes:
            nodes.append(i[1])
        # weight needed to work with current algorithm implementation - can be removed by editing how networks are read
        # in by the aList() class
        network.append((i[0], i[1], random.randint(1, max_weight)))
        # append reverse direction
        network.append((i[1], i[0], random.randint(1, max_weight)))
        temp = (i[1], i[0])
        if vtype.lower() == "d":
            # discrete
            alpha[i] = random.randint(2, 3)
        else:
            # continuous
            alpha[i] = random.uniform(1.1, 3)
        sigma[i] = random.uniform(1, (0.3 * k) ** alpha[i])
        qval[i] = sigma[i] ** (1 / alpha[i])
        alpha[temp] = alpha[i]
        sigma[temp] = sigma[i]
        qval[temp] = qval[i]
    j = 0
    while j < k:
        s = random.choice(nodes)
        t = random.choice(nodes)
        if s != t and (s, t) not in outputpairs:
            outputpairs.append((s, t))
            j += 1
    #
    # print("Nodes:", max_nodes)
    return network, outputpairs, nodes, alpha, qval, sigma

def networkGen3(k, edges, nodes, vtype="c"):
    # network = []
    # nodes = []
    alpha = {}
    sigma = {}
    # qval = {}
    outputpairs = []
    for i in edges:
        # networkGen3 takes in input for nodes used
        # if i[0] not in nodes:
        #     nodes.append(i[0])
        # if i[1] not in nodes:
        #     nodes.append(i[1])

        # weight needed to work with current algorithm implementation - can be removed by editing how networks are read
        # in by the aList() class
        # network.append((i[0], i[1], random.randint(1, max_weight)))
        # append reverse direction (disregard as considering undirected graph)
        # network.append((i[1], i[0], random.randint(1, max_weight)))
        temp = (i[1], i[0])
        if vtype.lower() == "d":
            # discrete
            alpha[i] = random.randint(2, 3)
        else:
            # continuous
            alpha[i] = random.uniform(1.1, 3)
        sigma[i] = random.uniform(1, (0.3 * k) ** alpha[i])
        # qval[i] = sigma[i] ** (1 / alpha[i])
        alpha[temp] = alpha[i]
        sigma[temp] = sigma[i]
        # qval[temp] = qval[i]
    j = 0
    while j < k:
        s = random.choice(nodes)
        t = random.choice(nodes)
        if s != t and (s, t) not in outputpairs:
            outputpairs.append((s, t))
            j += 1
    return outputpairs, nodes, alpha, sigma

def generateEdges(numNodes, numEdges):
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
    return nodes, edges

# network: abilene/nsf, vtype:continuous(c) or discrete(d) alpha values, k: num of pairs, num: instance #
# alpha, sigma, and pairs to be written to file
def fileGen(edges, nodes, networkname, vtype, k, num):
    # network, outputpairs, nodes, alpha, qval, sigma = networkGen2(k, edges, vtype)
    outputpairs, nodes, alpha, sigma = networkGen3(k, edges, nodes, vtype)
    string = networkname + "_" + vtype + "_k" + str(k) + "_" + str(num) + ".txt"

    # edit to write in nodes
    f = open(string, "w+")
    f.write(str(alpha) + "\n")
    # f.write(": Alpha\n")
    f.write(str(sigma) + "\n")
    # f.write(": Sigma\n")
    f.write(str(outputpairs) + "\n")
    # f.write(": Pairs")
    f.write(str(nodes) + "\n")
    f.close()

    print(string)
    return string

# reads from a file, compute
def fileRead(filename):
    name = filename.split("_")
    f = open(filename, "r")
    alpha = ast.literal_eval(f.readline())
    sigma = ast.literal_eval(f.readline())
    pairs = ast.literal_eval(f.readline())
    #nodes = ast.literal_eval(f.readline())
    nodes = []
    network = []
    qval = {}
    for i in alpha:
        if i[0] not in nodes:
            nodes.append(i[0])
        if i[1] not in nodes:
            nodes.append(i[1])
        # weight needed to work with current algorithm implementation - can be removed by editing how networks are read
        # in by the aList() class

        # ensure only done once
        # network.append((i[0], i[1], random.randint(1, max_weight)))
        network.append((i[0], i[1]))
        qval[i] = sigma[i] ** (1 / alpha[i])
    return network, pairs, nodes, alpha, qval, sigma


# calculates psi values for given load and weights for graph, alpha can be adjusted (default 2)
# sigma has to be added (and then formula changed)
def psicalc(graph, alpha, qval):
    for f_node in graph.edges:
        for t_node in graph.edges[f_node]:
            graph.psi[(f_node, t_node)] = (qval[(f_node, t_node)] ** (alpha[(f_node, t_node)] - 1)) * (1 + (1 / e)) \
                                          + (alpha[(f_node, t_node)] * (
                    graph.load[(f_node, t_node)] ** (alpha[(f_node, t_node)] - 1)))
            + ((alpha[(f_node, t_node)] ** alpha[(f_node, t_node)]) / e)
            # old psi calculation (without sigma values)
            # graph.psi[(f_node, t_node)] = (alpha * (graph.load[(f_node, t_node)] ** (alpha - 1)) +
            # ((alpha ** alpha) / e)) * graph.weights[(f_node, t_node)]


# takes in a pair, performs psi calculation, finds shortest path for pair and then updates graph.load
# returns path taken for storing
def algorithm(graph, pair, alpha, qval):
    psicalc(graph, alpha, qval)
    path = dijkstra2(graph, pair[0], pair[1])
    for i in range(len(path) - 1):
        graph.load[(path[i], path[i + 1])] += 1
        graph.load[(path[i+1], path[i])] += 1
    return path


# generates a network and pairs, continues to run until all pairs are completed
def algo_main(network, pairs, alpha, qval, sigma):
    x = aList()
    algo_sum = 0
    for edge in network:
        x.add_edge(*edge)
    for i in pairs:
        algorithm(x, i, alpha, qval)
    for j in alpha:
        #  algo_sum += x.weights[j] * (x.load[j] ** alpha[j])
        algo_sum += x.load[j] ** alpha[j]
        if x.load[j] >= 1:
            algo_sum += sigma[j]
    return algo_sum

# 50 nodes, 150 edges
# nodes, edges = generateEdges(50, 150)
# network, outputpairs, newnodes, alpha, qval, sigma = fileRead(fileGen(edges, nodes, "test", 'c', 25, 1))