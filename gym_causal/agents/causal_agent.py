import numpy as np
import networkx as nx
import random
import scipy.special as spc

class Agent(object):
    """
    Causal Agent
    """
    def __init__(self, act_space, obs_space, rew_space):

        self.action_space = act_space
        self.observation_space = obs_space
        self.reward_space = rew_space
        self.time = 0
        self.occ_table = np.zeros([len(self.reward_space),
                           len(self.action_space),
                           len(self.observation_space)])
        self.G = self.create_graph()
        self.memory = np.zeros((1, 4))

    def create_graph(self):
        """
        Init the causal graph

        """

        G = nx.DiGraph()

        for i in range(len(self.observation_space)):
            node_name = "obs" + str(i)
            G.add_node(node_name)
        for i in range(len(self.action_space)):
            node_name = "act" + str(i)
            G.add_node(node_name)
        for i in range(len(self.reward_space)):
            node_name = "rew" + str(i)
            G.add_node(node_name)

        return G

    def act(self, temp):

        if (self.time == 0):
            action = random.choice(self.action_space)
        else:
            probs = np.exp(self.occ_table[1:].sum(axis=0)/temp) / np.sum(np.exp(self.occ_table[1:].sum(axis=0)/temp))
            cumul = 0
            choice = random.uniform(0, 1)
            for a, pr in enumerate(probs[:, 1]):
                cumul += pr
                if cumul > choice:
                    action = a
                else:
                    action = random.choice(self.action_space)
        self.time += 1

        return action



    def update_memory(self, act, obs, rew, done):

        self.memory = np.append(self.memory, np.array([[act, obs, rew, done]]), axis = 0)

        # first get the nodes corresponding to last action/ob/rew
        action_node = "act" + str(np.where(act == self.action_space)[0][0])
        obs_node = "obs" + str(np.where(obs == self.observation_space)[0][0])
        rew_node = "rew" + str(np.where(rew == self.reward_space)[0][0])

        # check if the edges already exist to update the weights
        if ((action_node, obs_node) in self.G.edges):
            act_obs_weight = self.G[action_node][obs_node]['weight']
        else:
             act_obs_weight = 1

        if ((action_node, rew_node) in self.G.edges):
            act_rew_weight = self.G[action_node][rew_node]['weight']
        else:
            act_rew_weight = 1

        if ((obs_node, rew_node) in self.G.edges):
            obs_rew_weight = self.G[obs_node][rew_node]['weight']
        else:
            obs_rew_weight = 1

        # then draw edges from action to (obs and rew) nodes
        self.G.add_edge(action_node, obs_node, weight = act_obs_weight + 1)
        self.G.add_edge(action_node, rew_node, weight = act_rew_weight + 1)
        self.G.add_edge(obs_node, rew_node, weight = obs_rew_weight + 1)

        # update co occurence table
        i = (np.where(rew == self.reward_space)[0][0])
        j = (np.where(act == self.action_space)[0][0])
        k = (np.where(obs == self.observation_space)[0][0])
        self.occ_table[i][j][k] += 1


class RandomAgent(object):
    """
    Random agent
    """
    def __init__(self,
                 action_space,
                 observation_space,
                 reward_space):

        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_space = reward_space
        self.time = 0


    def act(self):

        action = random.choice(self.action_space)
        self.time += 1

        return action
