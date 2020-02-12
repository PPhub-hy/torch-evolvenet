# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:32:54 2019

@author: Hongyu Li
"""
from lib.graph_model import DAG
from lib import torch_EvolveNet
from lib.cifar10_dataset import CIFAR10_Dataset
from lib import analysis_connections

import torch.nn as nn
import time
import math
import torch
import os

def train_evolving_net(Evolving = False):
    
    savepath = "./models/"
    #新建DAG，存储模型结构
    dag = DAG(empty = True)
    model_name = '2019-12-23_11:13:32_rand_N512_edge8192_InOut3-11'
    
    #装载原DAG
    dag.load_structure(savepath, model_name)
    dag.limitation = 512
    
    #检查DAG结构
    print("********************The DAG*******************")
    print("input: ", dag.input_nodes)
    print("output: ", dag.output_nodes)
    print("pooling_gate: ", dag.pooling_gate)
    dag.sum_edges_num()
    #for node, connections in dag.nodes_connections.items():
    #    print(node, connections)
    #for node, connections in dag.connecting_output.items():
    #    print(node, connections)

    Reslink = 'None' # 'None'、'flow'、'stream'
    multi_gpu = False
    
    #初始化模型
    
    net = torch.load(savepath + model_name + '_model.pkl')
    
    print('Reslink = ', Reslink)
    print('parameters(M): ', '  Trainable:', net.get_parameter_number()['Trainable'] / 1e6, '  Total:', net.get_parameter_number()['Total'] / 1e6)
    print(torch.cuda.device_count(), 'GPU is used')
    if multi_gpu:
        net = nn.DataParallel(net.cuda())
        print("model ", net.module.network.name, " is structured")
    else:
        net = net.cuda()
        print("model ", net.network.name, " is structured")
    
    
    DATA_DIR = 'lib/CIFAR10_dataset'
    BatchSize_test = 100
    
    #加载数据
    cifar10 = CIFAR10_Dataset(data_dir = DATA_DIR)
            
    #检验模型
    accuracy = torch_EvolveNet.test_model(net, cifar10, BatchSize_test)
    print("final accuracy = {:.4f}".format(accuracy))
    if multi_gpu:
        with open(savepath + '/' + net.module.network.name + '_accuracys.txt', 'a+') as f:
            f.write("{:.4f}\n".format(accuracy))
    else:
        with open(savepath + '/' + net.network.name + '_accuracys.txt', 'a+') as f:
            f.write("{:.4f}\n".format(accuracy))
        
if __name__ == '__main__':
    train_evolving_net(Evolving = True)
    