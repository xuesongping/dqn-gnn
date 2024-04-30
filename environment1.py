# Copyright (c) 2021, Paul Almasan [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: felician.paul.almasan@upc.edu

import gym
import numpy as np
import networkx as nx
import random
from gym import error, spaces, utils
from random import choice
import pylab
import json
import gc
import matplotlib.pyplot as plt


def create_fujian_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    Gbase.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
         (10, 11), (11, 12), (12, 13), (13, 14), (14, 0), (14, 15), (15, 16), (16, 17), (17, 18), (18, 6),
         (4, 19), (6, 21), (6, 22), (20, 21), (21, 22), (22, 23), (14, 24), (24, 25), (25, 26), (24, 26), (13, 16),
         (13, 27), (27, 28), (12, 28), (11, 17), (9, 18), (7, 18), (1, 16), (16, 27), (27, 12)])

    return Gbase


plt.show()


def create_geant2_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (1, 3), (1, 6), (1, 9), (2, 3), (2, 4), (3, 6), (4, 7), (5, 3),
         (5, 8), (6, 9), (6, 8), (7, 11), (7, 8), (8, 11), (8, 20), (8, 17), (8, 18), (8, 12),
         (9, 10), (9, 13), (9, 12), (10, 13), (11, 20), (11, 14), (12, 13), (12, 19), (12, 21),
         (14, 15), (15, 16), (16, 17), (17, 18), (18, 21), (19, 23), (21, 22), (22, 23)])

    return Gbase


def create_nsfnet_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 8), (3, 4), (4, 5), (4, 6), (5, 12), (5, 13),
         (6, 7), (7, 10), (8, 9), (8, 11), (9, 10), (9, 12), (10, 11), (10, 13), (11, 12)])

    return Gbase


def create_small_top():
    G = nx.Graph()

    # 添加24个节点
    G.add_nodes_from(range(24))

    # 将节点连接成环形图
    edges = [(i, (i + 1) % 24) for i in range(24)]

    # 随机选择10条额外的边
    random_edges = random.sample(
        [(i, j) for i in range(24) for j in range(i + 1, 24) if (i, j) not in edges and (j, i) not in edges], 10)

    G.add_edges_from(edges + random_edges)

    return G


plt.show()


def create_gbn_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    Gbase.add_edges_from(
        [(0, 2), (0, 8), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 9), (4, 8), (4, 10), (4, 9),
         (5, 6), (5, 8), (6, 7), (7, 8), (7, 10), (9, 10), (9, 12), (10, 11), (10, 12), (11, 13),
         (12, 14), (12, 16), (13, 14), (14, 15), (15, 16)])

    return Gbase


def generate_nx_graph(topology):
    """
    Generate graphs for training with the same topology.
    """
    if topology == 0:
        G = create_nsfnet_graph()
    elif topology == 1:
        G = create_fujian_graph()
    elif topology == 2:
        G = create_small_top()
    elif topology == 3:
        G = create_gbn_graph()
    elif topology == 4:
        G = create_geant2_graph()

    # nx.draw(G, with_labels=True)
    # plt.show()
    # plt.clf()
    rand_num = random.random()
    # Node id counter
    incId = 1
    # Put all distance weights into edge attributes.
    for (i, j, attrs) in G.edges.data():
        attrs['edgeId'] = incId
        attrs['betweenness'] = 0
        attrs['numsp'] = 0  # Indicates the number of shortest paths going through the link
        # We set the edges capacities to 200
        if random.random() < 0.6:
            attrs['edgecapacity'] = 200
        else:
            attrs['edgecapacity'] = 400
        attrs['bw_allocated'] = 0

        if random.random() < 0.6:
            attrs['changdu'] = random.randint(10, 100)
        else:
            attrs['changdu'] = random.randint(100, 300)

        attrs['edgedependability'] = random.uniform(0.985, 1)
        incId = incId + 1
    """
    为图G中的每个节点设置dependability属性
    """
    edges_list = list(G.edges)
    rand = random.random()

    #if rand < 0.1:
    #selected_edge = random.sample(edges_list, 2)
    #for edge in selected_edge:
     #       G.edges[edge]['edgecapacity'] = 0
    #if rand < 0.5:
    #selected_edge = random.sample(edges_list, 1)
    #for edge in selected_edge:
     #   G.edges[edge]['edgecapacity'] = 0
      #  G.edges[edge]['edgedependability'] = 0
        #    G.edges[edge]['changdu'] = 1000

    #else:
     #   pass

    for node in G.nodes():
        # 设置节点的dependability属性
        if random.random() <= 0.4:
            G.nodes[node]['nodedependability'] = 0.985
            G.nodes[node]['nodecapacity'] = 1800
        else:
            G.nodes[node]['nodedependability'] = 0.99
            G.nodes[node]['nodecapacity'] = 2000
    return G


def compute_link_betweenness(g, k):
    n = len(g.nodes())
    betw = []
    for i, j in g.edges():
        # we add a very small number to avoid division by zero
        b_link = g.get_edge_data(i, j)['numsp'] / ((2.0 * n * (n - 1) * k) + 0.00000001)
        g.get_edge_data(i, j)['betweenness'] = b_link
        betw.append(b_link)

    mu_bet = np.mean(betw)
    std_bet = np.std(betw)
    return mu_bet, std_bet


class Env1(gym.Env):
    """
    Description:
    The self.graph_state stores the relevant features for the GNN model

    self.graph_state[:][0] = CAPACITY
    self.graph_state[:][1] = BW_ALLOCATED
  """

    def __init__(self):
        self.node_dependability = None
        self.edge_dependability = None
        self.edge_length = None
        self.graph = None
        self.initial_state = None
        self.source = None
        self.destination = None
        self.demand = None
        self.link_state = None
        self.node_state = None
        self.diameter = None
        self.state1 = None

        # Nx Graph where the nodes have features. Betweenness is allways normalized.
        # The other features are "raw" and are being normalized before prediction
        self.first = None
        self.firstTrueSize = None
        self.second = None
        self.between_feature = None

        # Mean and standard deviation of link betweenness
        self.mu_bet = None
        self.std_bet = None

        self.max_demand = 0
        self.K = 4
        self.listofDemands = None
        self.nodes = None
        self.nodesDict = None
        self.ordered_edges = None
        self.ordered_nodes = None
        self.edgesDict = None
        self.numNodes = None
        self.numEdges = None
        self.edgedependability_feature = None
        self.edgecapacity_feature = None
        self.changdu_feature = None
        self.nodecapacity_feature = None
        self.nodedependability_feature = None

        self.state = None
        self.episode_over = True
        self.reward = 0
        self.allPaths = dict()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def num_shortest_path(self, topology):
        self.diameter = nx.diameter(self.graph)

        # Iterate over all node1,node2 pairs from the graph
        for n1 in self.graph:
            for n2 in self.graph:
                if (n1 != n2):
                    # Check if we added the element of the matrix
                    if str(n1) + ':' + str(n2) not in self.allPaths:
                        self.allPaths[str(n1) + ':' + str(n2)] = []

                    # First we compute the shortest paths taking into account the diameter
                    # This is because large topologies might take too long to compute all shortest paths
                    [self.allPaths[str(n1) + ':' + str(n2)].append(p) for p in
                     nx.all_simple_paths(self.graph, source=n1, target=n2, cutoff=self.diameter * 2)]

                    # We take all the paths from n1 to n2 and we order them according to the path length
                    self.allPaths[str(n1) + ':' + str(n2)] = sorted(self.allPaths[str(n1) + ':' + str(n2)],
                                                                    key=lambda item: (len(item), item))

                    path = 0
                    while path < self.K and path < len(self.allPaths[str(n1) + ':' + str(n2)]):
                        currentPath = self.allPaths[str(n1) + ':' + str(n2)][path]
                        i = 0
                        j = 1

                        # Iterate over pairs of nodes increase the number of sp
                        while (j < len(currentPath)):
                            self.graph.get_edge_data(currentPath[i], currentPath[j])['numsp'] = \
                                self.graph.get_edge_data(currentPath[i], currentPath[j])['numsp'] + 1
                            i = i + 1
                            j = j + 1

                        path = path + 1

                    # Remove paths not needed
                    del self.allPaths[str(n1) + ':' + str(n2)][path:len(self.allPaths[str(n1) + ':' + str(n2)])]
                    gc.collect()

    def _first_second_between(self):
        self.first = list()
        self.second = list()

        # For each edge we iterate over all neighbour edges
        for i, j in self.ordered_edges:
            neighbour_edges = self.graph.edges(i)

            for m, n in neighbour_edges:
                if (i != m or j != n) and (i != n or j != m):
                    self.first.append(self.edgesDict[str(i) + ':' + str(j)])
                    self.second.append(self.edgesDict[str(m) + ':' + str(n)])

            neighbour_edges = self.graph.edges(j)
            for m, n in neighbour_edges:
                if (i != m or j != n) and (i != n or j != m):
                    self.first.append(self.edgesDict[str(i) + ':' + str(j)])
                    self.second.append(self.edgesDict[str(m) + ':' + str(n)])

    def generate_environment(self, topology, listofdemands):
        # The nx graph will only be used to convert graph from edges to nodes
        self.graph = generate_nx_graph(topology)

        self.listofDemands = listofdemands

        self.max_demand = np.amax(self.listofDemands)

        # Compute number of shortest paths per link. This will be used for the betweenness
        self.num_shortest_path(topology)

        # Compute the betweenness value for each link
        self.mu_bet, self.std_bet = compute_link_betweenness(self.graph, self.K)

        self.edgesDict = dict()
        self.nodesDict = dict()

        some_edges_1 = [tuple(sorted(edge)) for edge in self.graph.edges()]
        some_nodes_1 = self.graph.nodes()

        self.ordered_edges = sorted(some_edges_1)
        self.ordered_nodes = sorted(some_nodes_1)

        self.numNodes = len(self.graph.nodes())
        self.numEdges = len(self.graph.edges())

        self.link_state = np.zeros((self.numEdges, 4))
        self.node_state = np.zeros((self.numNodes, 2))
        self.between_feature = np.zeros(self.numEdges)

        position = 0

        for edge in self.ordered_edges:
            i = edge[0]
            j = edge[1]
            self.edgesDict[str(i) + ':' + str(j)] = position
            self.edgesDict[str(j) + ':' + str(i)] = position

            betweenness = (self.graph.get_edge_data(i, j)['betweenness'] - self.mu_bet) / self.std_bet
            self.graph.get_edge_data(i, j)['betweenness'] = betweenness
            self.link_state[position][0] = self.graph.edges[i, j]["edgecapacity"]
            self.link_state[position][2] = self.graph.edges[i, j]['edgedependability']
            self.link_state[position][3] = self.graph.edges[i, j]['changdu']
            self.between_feature[position] = self.graph.edges[i, j]['betweenness']
            position = position + 1

        for i in range(len(self.ordered_nodes)):
            self.nodesDict[str(self.ordered_nodes[i])] = i
            self.node_state[i][0] = self.graph.nodes[self.ordered_nodes[i]]['nodedependability']
            self.node_state[i][1] = self.graph.nodes[self.ordered_nodes[i]]['nodecapacity']

        if self.node_state.shape[0] < self.link_state.shape[0]:
            padded_node_state = np.pad(self.node_state,
                                       ((0, self.link_state.shape[0] - self.node_state.shape[0]), (0, 0)),
                                       mode='constant', constant_values=0)
        else:
            padded_node_state = self.node_state

        self.initial_state = np.concatenate((self.link_state, padded_node_state), axis=1)

        self._first_second_between()

        self.firstTrueSize = len(self.first)

        # We create the list of nodes ids to pick randomly from them
        self.nodes = list(range(0, self.numNodes))

    def make_step(self, state, action, demand, source, destination):
        self.graph_state = np.copy(state)
        self.episode_over = True
        self.reward = 0
        self.node_dependability = []
        self.edge_dependability = []
        self.edge_length = []
        i = 0
        j = 1
        currentPath = self.allPaths[str(source) + ':' + str(destination)][action]

        # Once we pick the action, we decrease the total edge capacity from the edges
        # from the allocated path (action path)
        while (j < len(currentPath)):
            self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] -= demand
            self.graph_state[self.nodesDict[str(currentPath[i])]][5] -= demand
            self.graph_state[self.nodesDict[str(currentPath[j])]][5] -= demand
            if ((self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] < 0)
                    or (self.graph_state[self.nodesDict[str(currentPath[i])]][5] < 0)
                    or (self.graph_state[self.nodesDict[str(currentPath[j])]][5] < 0)):
                self.reward = self.reward - demand / 64
                # FINISH IF LINKS CAPACI
                return self.graph_state, self.reward, self.episode_over, self.demand, self.source, self.destination

            i = i + 1
            j = j + 1

        # We update the node state
        i = 0
        j = 1
        while j < len(currentPath):
            # 提取节点可靠性
            self.node_dependability.append(self.graph_state[self.nodes[i]][4])

            # 提取边的可靠性
            edge = str(currentPath[i]) + ':' + str(currentPath[j])
            self.edge_dependability.append(self.graph_state[self.edgesDict[edge]][2])
            self.edge_length.append(self.graph_state[self.edgesDict[edge]][3])
            i = i + 1
            j = j + 1
        total_edge_length = 0
        for long in self.edge_length:
            total_edge_length += long
        # 计算节点可靠性和边的可靠性之积
        result = 1
        for node_dep, edge_dep in zip(self.node_dependability, self.edge_dependability):
            result *= node_dep * edge_dep

        if demand == 8:
            if result > 0.9 and total_edge_length < 1000:
                self.reward = self.reward + 8
            else:
                self.reward = self.reward - 8
                self.reward = self.reward / 64
                return (self.graph_state, self.reward, self.episode_over, self.demand, self.source,
                        self.destination)
        elif demand == 32:
            if result > 0.88 and total_edge_length < 950:
                self.reward = self.reward + 32
            else:
                self.reward = self.reward - 32
                self.reward = self.reward / 64
                return (self.graph_state, self.reward, self.episode_over, self.demand, self.source,
                        self.destination)
        else:
            if result > 0.86 and total_edge_length < 900:
                self.reward = self.reward + 64
            else:
                self.reward = self.reward - 64
                self.reward = self.reward / 64
                return (self.graph_state, self.reward, self.episode_over, self.demand, self.source,
                        self.destination)
        # Leave the bw_allocated back to 0
        self.graph_state[:, 1] = 0

        # Reward is the allocated demand or 0 otherwise (end of episode)
        # We normalize the demand to don't have extremely large values
        self.reward = self.reward / 64
        self.episode_over = False

        self.demand = random.choice(self.listofDemands)
        self.source = random.choice(self.nodes)

        # We pick a pair of SOURCE,DESTINATION different nodes
        while True:
            self.destination = random.choice(self.nodes)
            if self.destination != self.source:
                break

        return self.graph_state, self.reward, self.episode_over, self.demand, self.source, self.destination

    def reset(self):
        """
        Reset environment and setup for new episode. Generate new demand and pair source, destination.

        Returns:
            initial state of reset environment, a new demand and a source and destination node
        """
        self.graph_state = np.copy(self.initial_state)

        self.demand = random.choice(self.listofDemands)

        self.source = random.choice(self.nodes)
        while True:
            self.destination = random.choice(self.nodes)
            if self.destination != self.source:
                break

        return self.graph_state, self.demand, self.source, self.destination

    def eval_sap_reset(self, demand, source, destination):
        """
        Reset environment and setup for new episode. This function is used in the "evaluate_DQN.py" script.
        """
        self.graph_state = np.copy(self.initial_state)
        self.demand = demand
        self.source = source
        self.destination = destination

        return self.graph_state
