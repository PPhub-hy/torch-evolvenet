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

def train_evolving_net():
    '''装载原DAG'''
    model_path = "./models/"
    dag_loading_name = '2019-12-23_11:13:32_rand_N512_edge8192_InOut3-11'
    dag_loading_path = model_path
    savepath = model_path + dag_loading_name + '_sketch/'
    
    dag = DAG(empty = True)
    dag.load_structure(dag_loading_path, dag_loading_name)
        
    epoches = 100
    
    os.mkdir(savepath)
    
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
    
    Learning_Rate = 0.1
    lr_min = 1e-5
    Weight_Decay = 1e-4
    Momentum = 0.9
    dropout = 0
    classes = 10
    SCALE = 4
    Ext_ratio = 1
    Leaky_a = 0.1
    Reslink = 'None' # 'None'、'flow'、'stream'
    multi_gpu = False
    epoches_consine = epoches
    scheduler_step = [80, 140, 170]
    
    #初始化模型
    print('cosine from {} to {} in {} epoches'.format(Learning_Rate, lr_min, epoches_consine))
    net = torch_EvolveNet.Net(dag, classes+1, 
                              scale = SCALE, 
                              ext_ratio = Ext_ratio,
                              dropout_rate=dropout, 
                              learning_rate = Learning_Rate,
                              lr_min = lr_min,
                              mome = Momentum, 
                              w_d = Weight_Decay,
                              #scheduler_step = scheduler_step,
                              epoches = epoches_consine,
                              leaky_a = Leaky_a,
                              Res = Reslink)
    
    net.apply(torch_EvolveNet.weights_init)
    
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
    VAL_RATE = 0.1
    CUTOUT = True
    BatchSize = 64
    Iters = int(50000 * (1-VAL_RATE) // BatchSize)
    val_BatchSize = BatchSize
    BatchSize_test = BatchSize
    print_frequency = Iters
    
    #加载数据
    cifar10 = CIFAR10_Dataset(data_dir = DATA_DIR,
                              val_rate = VAL_RATE,
                              cutout = CUTOUT)
    # Construct batch-data generator
    train_generator = cifar10.batch_generator(BatchSize, split='train')
    val_generator = cifar10.batch_generator(BatchSize, split='val')
    
    #训练模型
    torch_EvolveNet.train_model(net, Iters, cifar10, train_generator, epoches, scheduler_step, BatchSize, val_BatchSize, print_frequency, logpath=savepath, multi_gpu = multi_gpu)
    epoches_consine -= epoches
    
    #保存模型参数文件
    torch.save(net, savepath + net.network.name + '_model.pkl')
    print("model saved at", savepath + net.network.name + '_model.pkl')
    
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
    train_evolving_net()
    