# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:00:48 2019

@author: anonymous
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import scipy.io as sio
from scipy.stats import chi2
import math
import os

from .graph_model import DAG
from . import torch_EvolveNet
from .cifar10_dataset import CIFAR10_Dataset

def DAG_iteration(net,
                  multi_gpu,
                  dag,
                  channels,
                  savepath, # an intermediate to save all link strength of every edge
                  iteration, # The number of iterations in DAG structure
                  reduce_num, # shows how many edges will be remove
                  extend_num, # shows how many edges will be relink
                  concentration,
                  dataset,
                  val_number,
                  Need_extend = True): # concentration of detected edges (bigger has smaller randomness)
    '''
    iterate DAG strcuture
    '''
    if multi_gpu:
        ext_ratio = net.module.ext_ratio
    else:
        ext_ratio = net.ext_ratio
    
    if multi_gpu:
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
        
    if multi_gpu:
        GFIR = net.module.GFIR
    else:
        GFIR = net.GFIR
    
    NODE_CHANNEL = channels
        
    if GFIR:
        importances = calculate_importances(net, multi_gpu, dataset, val_number, group_size = 100)
        weight_list = []
        for name, importance in importances.items():
            name = name.split(".")
            node_from = float(name[1].split("->")[0].replace('-', '.'))
            node_to = float(name[1].split("->")[1].replace('-', '.'))
            weight_list.append([importance, len(weight_list), node_from, node_to])
        
        norm_edges_list = estimate_edges_GFIR(weight_list,
                                              reduce_num = reduce_num, 
                                              concentration = concentration) 
    
    #save mat file for edges
    if not os.path.exists(savepath + dag.name + '.mat'):
        sio.savemat(savepath + dag.name + '.mat', {'norm_edges_list': norm_edges_list})
    
    #renew the dag according to the marks
    dag.renew(norm_edges_list, reduce_num, extend_num, Need_extend = Need_extend)
    
    #renew the parameters according to the new dag
    new_state, dag = renew_weight(state_dict, NODE_CHANNEL, ext_ratio, dag, Need_extend = Need_extend)
    
    return new_state, dag

def calculate_importances(net,
                         multi_gpu,
                         dataset,
                         val_number,
                         group_size):
    if multi_gpu:
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    
    importance = {}
    for name, parameter in state_dict.items():
        if name.split(".")[0] == 'param_masks':
            importance[name] = 0
    
    net.eval()
    #产生一个batch的检验数据
    print('calculating importances..........')
    val_generator = dataset.batch_generator(group_size, split='val')
    for idx in range(int(val_number)//group_size):
        val_batch = next(val_generator)
        #正向传递计算输出
        batch_input = torch.from_numpy(val_batch[0]).float()
        batch_input = batch_input.cuda()
        batch_output = net(batch_input)
        #计算损失函数
        batch_target = torch.Tensor(val_batch[1]).long()
        batch_target = batch_target.cuda()
        if multi_gpu:
            loss = net.module.criterion(batch_output, batch_target)
        else:
            loss = net.criterion(batch_output, batch_target)
        #梯度归零（否则累加）
        if multi_gpu:
            net.module.optimizer.zero_grad()
        else:
            net.optimizer.zero_grad()
        #反向传递计算梯度
        loss.backward()
        
        for name, parameter in net.named_parameters():
            if name.split(".")[0] == 'param_masks':
                if parameter.grad is None :
                    importance[name] += 0
                    continue
                imp = abs(parameter.cpu().detach().numpy()[0] * parameter.grad.cpu().numpy()[0])
                if float(name.split(".")[1].split("->")[1].replace('-', '.')) % 2 == 0:
                    imp = imp / 9
                importance[name] += imp
    print('importance calculated!')
    return importance

def estimate_edges_GFIR(edges_list, reduce_num, concentration):
    '''
    combine the norm of weight and grad, and return a final decision for every edge
    return a list[edges][weights, index, node_from, node_to, normed weights, decision(0 as the deletions)] #grads, #,normed grads, comperhensive score
    '''
    #calculate normed weight
    edges_list.sort(key = lambda x:x[0])
    for i in range(len(edges_list)):
        edges_list[i].append((edges_list[i][0]-edges_list[0][0])/(edges_list[-1][0]-edges_list[0][0]))
    
    norm_edges_list = edges_list
    #do decision
    norm_edges_list.sort(key = lambda x:x[4])
    for i in range(len(norm_edges_list)):
        norm_edges_list[i].append(1)
    
    for i in range(reduce_num):
        norm_edges_list[i][5]=0
        print('\redge {}->{} is delated'.format(norm_edges_list[i][2], norm_edges_list[i][3]), end= " ")
    print(' ')
    #rearrange all edges by their indexes
    norm_edges_list.sort(key = lambda x:x[2])
    
    return norm_edges_list

def renew_weight(trained_state, channels, ext_ratio, dag, Is_additional = False, Need_extend = False):
    '''
    renew the trained_state according to dag, and return a new one
    '''
    
    nodes_connections = dag.nodes_connections
    
    for node, connections in nodes_connections.items():
        if float(node) in dag.input_nodes:
            continue
        
        node_key = 'convs.' + node.replace('.','-') + '.weight'
        
        #逐个处理所有链接
        started = False
        chennel_shift = 0
        concat = 0
        for idx in range(len(connections)):
            while idx < len(connections) and type(connections[idx]) == tuple:
                this_channels = channels
                for num in range(dag.nodes_area[dag.all_nodes.index(connections[idx][1])]):
                    this_channels = math.ceil(this_channels * ext_ratio)
                if connections[idx][2] in dag.input_nodes or connections[idx][2] in dag.output_nodes:
                    this_channels = 1
                chennel_shift += this_channels
                del connections[idx]
            # if acheive the end of list stop the concat
            if idx == len(connections):
                break
            
            this_channels = channels
            for num in range(dag.nodes_area[dag.all_nodes.index(connections[idx])]):
                this_channels = math.ceil(this_channels * ext_ratio)
            if connections[idx] in dag.input_nodes or connections[idx] in dag.output_nodes :
                this_channels = 1

            GIFR_key = 'param_masks.' + str(connections[idx]).replace('.','-') + '->' + node.replace('.','-')
            
            if not started:
                if Need_extend:
                    concat = (trained_state[node_key][:, chennel_shift: chennel_shift + this_channels, :,:] * trained_state[GIFR_key][0]).cpu()
                else:
                    concat = trained_state[node_key][:, chennel_shift: chennel_shift + this_channels, :,:].cpu()
                started = True
            else:
                if Need_extend:
                    concat = torch.cat((concat, (trained_state[node_key][:, chennel_shift: chennel_shift + this_channels, :,:] * trained_state[GIFR_key][0]).cpu()), dim=1)
                else:
                    concat = torch.cat((concat, trained_state[node_key][:, chennel_shift: chennel_shift + this_channels, :,:].cpu()), dim=1)
            chennel_shift += this_channels
        # renew the paremeters with concat
        trained_state[node_key] = concat   
        
        if type(trained_state[node_key]) == int:
            continue
        
    return trained_state, dag

if __name__ == '__main__':
    dag_name = '2019-09-23_12:31:28_WS_N512_K16_p0.6'
    Savepath = 'models/' + dag_name + '/'
    
    parameter_file = Savepath + dag_name + '_parameters.pkl'
    dag_model = DAG()
    dag_model.load_structure(Savepath, dag_name)
    
    Trained_state = torch.load(parameter_file)
    
    
    REDUCE_RATIO = 0.2
    CONCENTRATION = 2
    Current_iteration = 0
    
    new_state, dag = DAG_iteration(dag_model, #model
                                   trained_state = Trained_state,
                                   savepath = Savepath, # an intermediate to save all link strength of every edge
                                   iteration = Current_iteration, # The number of iterations in DAG structure
                                   reduce_ratio = REDUCE_RATIO, # shows how many edges will be renew
                                   concentration = CONCENTRATION) # concentration of detected edges (bigger has smaller randomness)
    
    new_net = torch_EvolveNet.Net(dag, 11, 4, ext_ratio=1)
    new_net.load_state_dict(new_state)
    new_net = new_net.cuda()
    
    savepath = "./models/"
    DATA_DIR = 'lib/CIFAR10_dataset'
    VAL_RATE = 0.1
    epoches = 9
    print_frequency = 50
    BatchSize = 64
    Iters = int(50000 * (1-VAL_RATE) // BatchSize)
    cifar10 = CIFAR10_Dataset(data_dir=DATA_DIR,
                              val_rate=VAL_RATE)
    
    #训练模型
    torch_EvolveNet.train_model(new_net, Iters, cifar10, epoches, BatchSize, print_frequency, logpath=savepath)
    