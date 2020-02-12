# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:37:07 2019

@author: anonymous
"""
import pickle
import numpy as np
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DAG:    
    def __init__(self, nodes_num=5, #number of nodes
                 edges = 1000, # number of edges
                 in_nodes = 3, # input nodes number
                 out_nodes = 11,
                 adjoin=1 , p=0 , # parameters of WS model
                 name=None, # name of the model
                 clear_multiedge = True, # whether to clean the multiedges in WS model
                 local_degree = 10,
                 empty = False): 
        '''
        模型初始化（WS style）
        '''
        if empty:
            return
        self.name = name
        self.local_degree = local_degree

        self.init_rand_model(nodes_num, in_nodes, out_nodes, edges)#, rolling_range
        
        #所有节点名所处的空间分辨率区域
        self.nodes_area = []
        for i in range(len(self.all_nodes)):
            nodes_area = 0
            for j in range (len(self.pooling_gate)):
                if self.all_nodes[i] > self.pooling_gate[j]:
                    nodes_area += 1
            self.nodes_area.append(nodes_area)
            
        #节点输出指向字典
        self.renew_output()
        self.max_node = nodes_num - 1

    def init_rand_model(self, nodes_num, in_nodes, out_nodes, edges):#, rolling_range
        
        self.nodes_connections = {}
        self.input_nodes = []
        self.output_nodes = []
        self.pooling_gate = [nodes_num // 3 + 0.5, nodes_num * 2 // 3 + 0.5]
        
        # init all nodes
        for idx in range(nodes_num):
            self.nodes_connections[str(float(idx))] = []
            if idx < in_nodes:
                self.input_nodes.append(float(idx))
            if idx >= nodes_num - out_nodes:
                self.output_nodes.append(float(idx))
        
        #所有节点名集合，按数值大小排序
        self.all_nodes = []
        for node in self.nodes_connections.keys():
            self.all_nodes.append(float(node))
        self.all_nodes.sort()
        
        #模型更新时，新连接的起始索引
        self.renewing_connections = {}
        for node in self.all_nodes:
            self.renewing_connections[str(node)]=[]
        
        # link
        #self.limitation = int(nodes_num * rolling_range)
        added_num = self.complete_DAG(Is_init = True)
        self.combine_connections()
        self.renew_output()
        for i in range(edges - added_num):
            randh_index, randl_index = self.rand_edge(Is_local = True)
            self.nodes_connections[str(self.all_nodes[randh_index])].append(float(self.all_nodes[randl_index]))

    def rand_edge(self, Is_local):
        nodelist = self.all_nodes
        
        randh_index = -1
        randl_index = -1
        while randh_index == -1 \
            or float(nodelist[randh_index]) in self.input_nodes \
            or float(nodelist[randl_index]) in self.output_nodes \
            or float(nodelist[randl_index]) in self.nodes_connections[str(nodelist[randh_index])] \
            or float(nodelist[randl_index]) in self.renewing_connections[str(nodelist[randh_index])] \
            or randh_index == randl_index \
            or len(self.connecting_output[str(nodelist[randh_index])]) + len(self.nodes_connections[str(nodelist[randh_index])]) == 0 \
            or len(self.connecting_output[str(nodelist[randl_index])]) + len(self.nodes_connections[str(nodelist[randl_index])]) == 0: #\
            #or nodelist[randh_index] - nodelist[randl_index] > self.limitation:
            randh_index = np.random.randint(0,len(nodelist)-1)
            randl_index = np.random.randint(0,len(nodelist)-1)
            if Is_local:
                Pass = False
                times = self.local_degree
                while not Pass:
                    randh_index = np.random.randint(0,len(nodelist)-1)
                    randl_index = np.random.randint(0,len(nodelist)-1)
                    length = abs(randh_index - randl_index)
                    for i in range(times):
                        rand_test = np.random.randint(0,len(nodelist)-1)
                        if rand_test > length:
                            if i == times-1:
                                Pass = True
                        else:
                            break
            else:
                randh_index = np.random.randint(0,len(nodelist)-1)
                randl_index = np.random.randint(0,len(nodelist)-1)
            if nodelist[randh_index] < nodelist[randl_index]:
                temp = randh_index
                randh_index = randl_index
                randl_index = temp
                
        assert float(nodelist[randh_index]) > float(nodelist[randl_index])
        assert float(nodelist[randl_index]) not in self.nodes_connections[str(nodelist[randh_index])]    
            
        return randh_index,randl_index

    def save_structure(self, savepath):
        '''
        保存模型为.pkl
        '''
        savemodel = {}
        savemodel["nodes"] = self.nodes_connections
        savemodel["pooling_gate"] = self.pooling_gate
        savemodel["input_nodes"] = self.input_nodes
        savemodel["output_nodes"] = self.output_nodes
        savemodel["max_node"] = self.max_node
        savemodel["name"] = self.name
        savemodel["all_nodes"] = self.all_nodes
        savemodel["nodes_area"] = self.nodes_area
        savemodel["renewing_connections"] = self.renewing_connections
        
        with open(savepath + self.name + '.pkl', 'wb') as f:
            pickle.dump(savemodel, f, pickle.HIGHEST_PROTOCOL)
            
        print("DAG saved as " + savepath + self.name + '.pkl')
    
    def load_structure(self, savepath, name):
        '''
        从.pkl读取模型
        '''
        with open(savepath + name + '.pkl', 'rb') as f:
            savemodel = pickle.load(f)

        self.name = savemodel["name"]
        self.nodes_connections = savemodel["nodes"] 
        self.pooling_gate = savemodel["pooling_gate"]
        self.input_nodes = savemodel["input_nodes"]
        self.output_nodes = savemodel["output_nodes"]
        self.max_node = savemodel["max_node"]
        self.all_nodes = savemodel["all_nodes"]
        self.nodes_area = savemodel["nodes_area"]
        if "renewing_connections" in savemodel.keys():
            self.renewing_connections = savemodel["renewing_connections"]
        else:
            print("WARNING!!!! no 'renewing_connections' in loaded model")
        
        print("DAG loaded: " + self.name)
        print("from: " + savepath)
    
    def renew_output(self):
        '''
        更新节点的输出指向字典
        '''
        self.connecting_output = {}
        for node in self.all_nodes:
            self.connecting_output[str(node)]=[]
        for node, connections in self.nodes_connections.items():
            for connect in connections:
                if type(connect) != tuple:
                    self.connecting_output[str(connect)].append(float(node))
        for node, connections in self.renewing_connections.items():
            for connect in connections:
                if type(connect) != tuple:
                    self.connecting_output[str(connect)].append(float(node))
    
    def renew(self, norm_edges_list, reduce_num, extend_num, Is_additional = False, Need_extend = True):
        delated_num = self.eliminate_edges(norm_edges_list, reduce_num, Is_additional)
        inserting_num = extend_num
        if Need_extend and not Is_additional:
            self.renew_output()
            self.extend_edges(inserting_num, Is_additional)
        self.renew_output()
        
        
    def eliminate_edges(self, norm_edges_list, reduce_num, Is_additional):
        '''
        eliminate selected edges in DAG
        norm_edges_list: list[edges][weights, index, node_from, node_to, normed weights, decision(0 as the deletions)] 
        '''
        self.sum_edges_num()
        if Is_additional:
            nodes_connections = self.renewing_connections
        else:
            nodes_connections = self.nodes_connections
            
        delated_num = 0
        for norm_edge in norm_edges_list:
            if norm_edge[5] == 0:
                remove_idx = nodes_connections[str(norm_edge[3])].index(norm_edge[2])
                removing_channels = self.nodes_area[self.all_nodes.index(nodes_connections[str(norm_edge[3])][remove_idx])]
                nodes_connections[str(norm_edge[3])][remove_idx] = (None, removing_channels, nodes_connections[str(norm_edge[3])][remove_idx])
                delated_num += 1
        
        assert delated_num == reduce_num
        print('{} edges are removed'.format(delated_num))
        
        self.sum_edges_num()
    
        return delated_num
        #for i in range(node_num): #num_nodes for weights
        #return weights
    
    def sum_edges_num(self):
        '''
        get edge number of DAG
        '''
        edge_num = 0
        for node, connections in self.nodes_connections.items():
            edge_num += len(connections)
            for connection in connections:
                if type(connection) == tuple:
                    edge_num -= 1
        
        edge_add_num = 0
        for node, connections in self.renewing_connections.items():
            edge_add_num += len(connections)
            for connection in connections:
                if type(connection) == tuple:
                    edge_add_num -= 1
        
        print("the DAG has {} edges now ({} + {}).".format(edge_num + edge_add_num, edge_num, edge_add_num))
        return edge_num, edge_add_num
    
    def complete_DAG(self, Is_additional = False, state_dict = None, channel = None, Is_init = False):
        '''
        check the connectivity of DAG and complete it to ensure every node has input and output
        '''
        added_num = 0
        nodelist = self.all_nodes
        
        for node in nodelist:
            if float(node) in self.input_nodes:
                continue
            
            #temply eliminate None elements
            connections = copy.deepcopy(self.nodes_connections[str(node)])
            for idx in range(len(connections)):
                while idx < len(connections) and type(connections[idx]) == tuple:
                    del connections[idx]
            connections_additional = copy.deepcopy(self.renewing_connections[str(node)])
            for idx in range(len(connections_additional)):
                while idx < len(connections_additional) and type(connections_additional[idx]) == tuple:
                    del connections_additional[idx]
                    
            # add an input if there's no input
            if len(connections) == 0 and len(connections_additional) == 0:
                #abandon isolatede nodes
                if not Is_init and len(self.connecting_output[str(node)]) == 0:
                    continue
                
                rand = -1
                while rand == -1 \
                    or nodelist[rand] >= float(node) \
                    or (not Is_init and len(self.nodes_connections[str(nodelist[rand])]) == 0): 
                    rand = np.random.randint(0,len(nodelist)-1)
                    
                self.renewing_connections[str(node)].append(nodelist[rand])
                print('\rinput added for node {}'.format(node),end= " ")
                added_num += 1
                    
        print(' ')
        for node in nodelist:
            if float(node) in self.output_nodes:
                continue

            #temply eliminate None elements
            connections = copy.deepcopy(self.nodes_connections[str(node)])
            for idx in range(len(connections)):
                while idx < len(connections) and type(connections[idx]) == tuple:
                    del connections[idx]
            connections_additional = copy.deepcopy(self.renewing_connections[str(node)])
            for idx in range(len(connections_additional)):
                while idx < len(connections_additional) and type(connections_additional[idx]) == tuple:
                    del connections_additional[idx]
            
            # check if there's no output
            sign = True
            for i in range(len(nodelist)):
                if float(node) in self.nodes_connections[str(nodelist[i])] \
                or float(node) in self.renewing_connections[str(nodelist[i])]:
                    sign = False
            
            #abandon isolatede nodes
            if not Is_init and len(connections) == 0 and len(connections_additional) == 0:
                sign = False
            
            # add an output if there's no output
            if sign:
                rand = -1
                while rand == -1 \
                    or nodelist[rand] <= float(node) \
                    or (not Is_init and len(self.connecting_output[str(nodelist[rand])]) == 0): 
                    rand = np.random.randint(nodelist.index(node), len(nodelist))
                self.renewing_connections[str(nodelist[rand])].append(node)
                print('\routput added for node {}'.format(node),end= " ")
                added_num += 1
                    
        print(' ')
        print('{} edges are added for complete the DAG'.format(added_num))
        print('DAG completed.'.format(added_num))
        self.sum_edges_num()
        
        return added_num

    
    def extend_edges(self, inserting_num, Is_additional):
        '''
        extend a reduced DAG to its original size
        '''
        #complete the DAG if any node has no input or output
        added_num = self.complete_DAG()
        remaining_num = inserting_num - added_num
        nodelist = [float(i) for i in list(self.nodes_connections.keys())]
        
        for i in range(remaining_num):
            randh_index, randl_index = self.rand_edge(Is_local = True)
            
            self.renewing_connections[str(nodelist[randh_index])].append(float(nodelist[randl_index]))
            print('\redge {}->{} is added'.format(float(nodelist[randl_index]), float(nodelist[randh_index])),end= " ")

        print(' ')
        self.complete_DAG()
        self.sum_edges_num()
   
    def combine_connections(self):
        '''
        combine all new connections to the general list
        '''
        for node in self.all_nodes:
            for connect in self.renewing_connections[str(node)]:
                self.nodes_connections[str(node)].append(connect)
            self.renewing_connections[str(node)] = []