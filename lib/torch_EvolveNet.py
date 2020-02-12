# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:36:15 2019

@author: anonymous
"""

import math
import time
import numpy as np
import os
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from lib.cifar10_dataset import CIFAR10_Dataset
from lib.graph_model import DAG

class Net(nn.Module):
    def __init__(self, dag, classes, scale,
                 ext_ratio = 1, 
                 dropout_rate = 0.5, 
                 learning_rate = 0.1,
                 lr_min = 0,
                 last_epoch = -1,
                 freeze_ratio = 0.1,
                 mome = 0.9, 
                 w_d = 4e-4,
                 scheduler_step = 1,
                 epoches = 1,
                 leaky_a = 0.1,
                 Res = 'None',
                 GFIR = False):
        super().__init__()
        self.convs = nn.ModuleDict()
        self.convs_add = nn.ModuleDict()
        #self.node_conv_acts = nn.ModuleDict()
        self.node_out_acts = nn.ModuleDict()
        self.network = dag
        self.ext_ratio = ext_ratio
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.momentum = mome
        self.weight_decay = w_d
        self.Res = Res
        self.GFIR = GFIR
        self.channels = scale

        '''
        all conv layers for nodes
        '''
        for str_node, connections in dag.nodes_connections.items():
            #根据结点位置确定当前结点粒度
            node_channel = scale
            for i in range(len(dag.pooling_gate)):
                if float(str_node) > dag.pooling_gate[i]:
                    node_channel = math.ceil(node_channel * ext_ratio)
            if float(str_node) in self.network.output_nodes or float(str_node) in self.network.input_nodes:
                node_channel = 1

            #节点输出激活
            node_out_act = nn.Sequential(nn.BatchNorm2d(node_channel, track_running_stats = True),# affine=False),
                                         #nn.LeakyReLU(negative_slope = leaky_a,inplace=True)
                                         #nn.ReLU(inplace = True)
                                         nn.PReLU(node_channel)
                                         )
            self.node_out_acts.add_module(str_node.replace('.','-') + '_out_act', node_out_act)
            
            if float(str_node) in self.network.input_nodes:
                continue
            
            if len(connections) > 0:
                #统计输入通道数
                input_channel = 0
                for connect in connections:
                    #保证其单向性
                    assert connect < float(str_node) 
                    #统计其他输入结点
                    this_input_channel = scale
                    if connect in self.network.input_nodes or connect in self.network.output_nodes:
                        this_input_channel = 1
                    for i in range(len(dag.pooling_gate)):
                        if connect > dag.pooling_gate[i]:
                            this_input_channel = math.ceil(this_input_channel * ext_ratio)
                    input_channel += this_input_channel
                #边变换                
                #if int(float(str_node)) % 20 == 19:
                    #conv = nn.Conv2d(in_channels=input_channel, out_channels=node_channel,  kernel_size=(3,3), padding=2, dilation=2, bias=False)
                if int(float(str_node)) % 2 == 0:
                    conv = nn.Conv2d(in_channels=input_channel, out_channels=node_channel,  kernel_size=(3,3), padding=1, bias=False)
                else:#elif int(float(str_node)) % 2 == 1:
                    conv = nn.Conv2d(in_channels=input_channel, out_channels=node_channel,  kernel_size=(1,1), padding=0, bias=False)
                
                self.convs.add_module(str_node.replace('.','-'), conv)
            
            if  len(dag.renewing_connections[str_node]) > 0:
                #统计输入通道数
                input_channel = 0
                for connect in dag.renewing_connections[str_node]:
                    #check the direction
                    assert connect < float(str_node) 
                    #calculate all the input channels for one node
                    this_input_channel = scale
                    if connect in self.network.input_nodes or connect in self.network.output_nodes:
                        this_input_channel = 1
                    for i in range(len(dag.pooling_gate)):
                        if connect > dag.pooling_gate[i]:
                            this_input_channel = math.ceil(this_input_channel * ext_ratio)
                    input_channel += this_input_channel
                
                #边变换-新加入的部分
                #if int(float(str_node)) % 20 == 19:
                #    conv = nn.Conv2d(in_channels=input_channel, out_channels=node_channel,  kernel_size=(3,3), padding=2, dilation=2, bias=False)
                if int(float(str_node)) % 2 == 0:
                    conv = nn.Conv2d(in_channels=input_channel, out_channels=node_channel,  kernel_size=(3,3), padding=1, bias=False)
                else:#elif int(float(str_node)) % 2 == 1:
                    conv = nn.Conv2d(in_channels=input_channel, out_channels=node_channel,  kernel_size=(1,1), padding=0, bias=False)
                
                self.convs_add.add_module(str_node.replace('.','-') + '_additional', conv)
        
        '''
        loss, optimizers
        '''
        self.criterion = nn.CrossEntropyLoss()
        if self.GFIR:
            param_mask = collections.OrderedDict()
            for node, connections in self.network.nodes_connections.items():
                for connect in connections:
                    param_name = str(connect).replace('.','-') + '->' + node.replace('.','-')
                    param_mask[param_name] = nn.Parameter(torch.randn(1) * 0.1 + 1)
            self.param_masks = nn.ParameterDict(param_mask)
            
            self.optimizer = optim.SGD(self.param_masks.parameters(), lr=learning_rate, momentum=mome, weight_decay=w_d)
        else:
            if len(self.convs_add) == 0:
                self.optimizer = optim.SGD([{'params': self.parameters(), 'initial_lr': learning_rate}], lr=learning_rate, momentum=mome, weight_decay=w_d)
                print('network with no additional connections!!!')
            else:
                self.optimizer = optim.SGD([
                                        {'params': self.convs.parameters()},
                                        {'params': self.node_out_acts.parameters()},
                                        {'params': self.convs_add.parameters(), 'lr': learning_rate}
                                        ], lr=learning_rate*freeze_ratio, momentum=mome, weight_decay=w_d)
                print('network with additional connections!!!')
        
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=0.2)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epoches, eta_min = lr_min, last_epoch = last_epoch)
    
    def forward(self, input):
        '''
        卷积部分的前向传播
        '''
        node_tensors = {}
        node_tensors_acted = {}
        
        all_nodes = self.network.all_nodes
        nodes_area = self.network.nodes_area
        connections = self.network.nodes_connections
        connections_add = self.network.renewing_connections

        #输入节点(越过池化门则池化)
        color = 0
        for input_node in self.network.input_nodes:
            temp_pool = input[:,color].unsqueeze(1)
            color +=1
            for i in range(nodes_area[all_nodes.index(input_node)]):
                temp_pool = F.avg_pool2d(temp_pool, kernel_size=(2,2))
            node_tensors[str(float(input_node))] = temp_pool
            node_tensors_acted[str(float(input_node))] = self.node_out_acts[str(float(input_node)).replace('.','-') + '_out_act'](node_tensors[str(float(input_node))])
        
        current_stage = 0
        #定义每一个节点的卷积层
        for i in range(len(all_nodes)):
            if all_nodes[i] in self.network.input_nodes:
                continue
            
            if len(connections[str(all_nodes[i])]) + len(connections_add[str(all_nodes[i])]) == 0:
                if all_nodes[i] in self.network.output_nodes:
                    node_tensors[str(all_nodes[i])] = torch.zeros(input.shape[0], 1, int(32 * 0.5 ** nodes_area[i]), int(32 * 0.5 ** nodes_area[i])).cuda()
                else:
                    node_tensors[str(all_nodes[i])] = torch.zeros(input.shape[0], self.channels, int(32 * 0.5 ** nodes_area[i]), int(32 * 0.5 ** nodes_area[i])).cuda()
            
            # pooling gate
            if current_stage < len(self.network.pooling_gate) and all_nodes[i] > self.network.pooling_gate[current_stage]:
                current_stage = current_stage + 1
                for k in node_tensors_acted.keys():
                    node_tensors[k] = F.avg_pool2d(node_tensors[k], kernel_size=(2,2))
                    node_tensors_acted[k] = F.avg_pool2d(node_tensors_acted[k], kernel_size=(2,2))
            
            '''
            当前节点的输入链接合并张量
            '''
            #拼接本体
            if len(connections[str(all_nodes[i])]) > 0:
                concat = node_tensors_acted[str(connections[str(all_nodes[i])][0])]
                if self.GFIR:
                    concat = concat * self.param_masks[str(connections[str(all_nodes[i])][0]).replace('.','-') + '->' + str(all_nodes[i]).replace('.','-')]
                    
            #拼接输入张量-原始结构
            for connect in connections[str(all_nodes[i])]:
                if len(connections[str(all_nodes[i])]) == 0 or connect == connections[str(all_nodes[i])][0]:
                    continue
                temp_input = node_tensors_acted[str(connect)]
                if self.GFIR:
                    temp_input = temp_input * self.param_masks[str(connect).replace('.','-') + '->' + str(all_nodes[i]).replace('.','-')]
                concat = torch.cat((concat, temp_input), dim=1)
                
            #拼接本体-新结构
            if len(connections_add[str(all_nodes[i])]) > 0:
                concat_add = node_tensors_acted[str(connections_add[str(all_nodes[i])][0])]

            #拼接输入张量-新结构     
            for connect in connections_add[str(all_nodes[i])]:
                if len(connections_add[str(all_nodes[i])]) == 0 or connect == connections_add[str(all_nodes[i])][0]:
                    continue
                temp_input = node_tensors_acted[str(connect)]
                concat_add = torch.cat((concat_add, temp_input), dim=1)

            if len(connections[str(all_nodes[i])]) > 0 and len(connections_add[str(all_nodes[i])]) > 0:
                node_tensors[str(all_nodes[i])] = self.convs[str(all_nodes[i]).replace('.','-')](concat) + self.convs_add[str(all_nodes[i]).replace('.','-') + '_additional'](concat_add)
            elif len(connections[str(all_nodes[i])]) > 0:
                node_tensors[str(all_nodes[i])] = self.convs[str(all_nodes[i]).replace('.','-')](concat)
            elif len(connections_add[str(all_nodes[i])]) > 0:
                node_tensors[str(all_nodes[i])] = self.convs_add[str(all_nodes[i]).replace('.','-') + '_additional'](concat_add)  
            
            node_tensors_acted[str(all_nodes[i])] = self.node_out_acts[str(all_nodes[i]).replace('.','-') + '_out_act'](node_tensors[str(all_nodes[i])])
            
        #输出节点
        concat = node_tensors_acted[str(float(self.network.output_nodes[0]))]
        for output_node in self.network.output_nodes:
            if output_node == self.network.output_nodes[0]:
                continue
            concat = torch.cat((concat, node_tensors_acted[str(float(output_node))]), dim=1)
        '''
        global avgpooling
        '''
        pool_size = (input.shape[2] // 2**len(self.network.pooling_gate), input.shape[3] // 2**len(self.network.pooling_gate))
        features = F.avg_pool2d(concat, kernel_size=pool_size)
        features = torch.squeeze(features,3)
        features = torch.squeeze(features,2)
        features = F.softmax(features, dim=1)
        
        return features  
    
    def reset_opimizer(self):
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
    
    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

def weights_init(m, leaky_a=0.1):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=leaky_a)

def calculate_accuracy(out, tar, batch_size):
    assert len(tar) == batch_size
    preds = out.argmax(dim=1)
    accuracy=[]
    for i in range(batch_size):
        if preds[i] == tar[i]:
            accuracy.append(1)
        else:
            accuracy.append(0)
    accuracy = torch.mean(torch.tensor(accuracy).float())
    return accuracy
    
def train_step(net, dataset, data_generator, batchsize, multi_gpu, print_needed=False, val_size=None, logpath=None):
    val_generator = dataset.batch_generator(batchsize, split='val')
    net.train()
    #产生一个batch的训练数据
    train_batch = next(data_generator)
    #正向传递计算输出
    batch_input = torch.from_numpy(train_batch[0]).float()
    batch_input = batch_input.cuda()
    batch_output = net(batch_input)
    #计算损失函数
    batch_target = torch.Tensor(train_batch[1]).long()
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
    #按照当前的梯度更新一轮变量
    if multi_gpu:
        nn.utils.clip_grad_norm_(net.module.parameters(), 5)
        net.module.optimizer.step()
    else:
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        net.optimizer.step()
    
    if print_needed:
        train_accuracy = calculate_accuracy(batch_output, batch_target, batchsize)
        net.eval()

        val_accuracy = []
        for i in range(5000//val_size):
            #产生一个batch的验证数据
            val_batch = next(val_generator)
            val_input = torch.from_numpy(val_batch[0]).float()
            val_input = val_input.cuda()
            with torch.no_grad():
                val_output = net(val_input)
            
            val_target = torch.Tensor(val_batch[1]).long()
            val_target = val_target.cuda()
            val_accuracy.append(calculate_accuracy(val_output, val_target, val_size))
        mean_val_accuracy = np.mean(val_accuracy)
        
        #记录log
        if multi_gpu:
            with open(logpath + '/' + net.module.network.name + '_course.txt', 'a+') as f:
                f.write("{:.3f}\t{:.3f}\t{:.3f}\n".format(loss.item(), train_accuracy, mean_val_accuracy))
        else:
            with open(logpath + '/' + net.network.name + '_course.txt', 'a+') as f:
                f.write("{:.3f}\t{:.3f}\t{:.3f}\n".format(loss.item(), train_accuracy, mean_val_accuracy))
        
        print('>>> train_loss: {:.3f}'.format(loss.item()))
        print('>>> train_accuacy: {:.3f}'.format(train_accuracy))
        print('>>> val_accuacy: {:.3f}'.format(mean_val_accuracy))

        net.train()
      
def train_model(net, iters, dataset, data_generator, epoches, scheduler_step, BatchSize, val_BatchSize, print_frequency, logpath, multi_gpu):
    if multi_gpu:
        if not os.path.exists(logpath + '/' + net.module.network.name + '_course.txt'):
            with open(logpath + '/' + net.module.network.name + '_course.txt', 'w+') as f:
                f.write("epoch\tstep\tloss\ttrain_acc\tval_accuracy\n")
        net.module.scheduler.step()
    else:
        if not os.path.exists(logpath + '/' + net.network.name + '_course.txt'):
            with open(logpath + '/' + net.network.name + '_course.txt', 'w+') as f:
                f.write("epoch\tstep\tloss\ttrain_acc\tval_accuracy\n")
        net.scheduler.step()
    
    for epoch in range(epoches):
        #if epoch in scheduler_step:
        if multi_gpu:
            net.module.scheduler.step()
        else:
            net.scheduler.step()
        print("=======================")
        if multi_gpu:
            print('epoch: ', epoch, 'lr: ', net.module.scheduler.get_lr())
        else:
            print('epoch: ', epoch + 1, 'lr: ', net.scheduler.get_lr())
        time_cost=[]
        for i in range(iters):
            if i%print_frequency==print_frequency-1:
                #记录log
                if multi_gpu:
                    with open(logpath + '/' + net.module.network.name + '_course.txt', 'a+') as f:
                        f.write("{}\t{}\t".format(epoch + 1, i))
                else:
                    with open(logpath + '/' + net.network.name + '_course.txt', 'a+') as f:
                        f.write("{}\t{}\t".format(epoch + 1, i))
                    
                #实时打印
                print("-----------------")
                print('epoch {0}/{1}, step {2}/{3}:'.format(epoch + 1, epoches, i, iters))
                train_step(net, dataset, data_generator, BatchSize, multi_gpu, print_needed=True, val_size=val_BatchSize, logpath=logpath)
                #计算用时
                if multi_gpu:
                    for param_group in net.module.optimizer.param_groups:
                        print(">>> lr: {:.4f}".format(param_group['lr']))
                        break
                else:
                    for param_group in net.optimizer.param_groups:
                        print(">>> lr: {:.4f}".format(param_group['lr']))
                        break
                print("cost {:.3f}s per iter".format(np.mean(time_cost)))
            else:
                start_time = time.time()
                train_step(net, dataset, data_generator, BatchSize, multi_gpu)
                time_cost.append(time.time() - start_time)
    

def test_model(net, dataset, batch_size):
    net.eval()
    test_datas, test_labels = dataset.get_test_data()
    total_imgs = len(test_labels)
    print("test data loaded: {} samples".format(total_imgs))
    
    test_datas = torch.tensor(test_datas).float()
    test_datas = test_datas.cuda()
    test_labels = torch.tensor(test_labels).long()
    test_labels = test_labels.cuda()
    print("model testing.................")
    #warm up
    for i in range(20):
        net(test_datas[:batch_size].cuda())
    print("model warmed.")
    
    accuracy = []
    for i in range(int(total_imgs//batch_size)):
        batch_input = test_datas[i * batch_size : (i+1) * batch_size]
        batch_labels = test_labels[i * batch_size : (i+1) * batch_size]
        with torch.no_grad():
            batch_output = net(batch_input)
            accuracy.append(calculate_accuracy(batch_output, batch_labels, batch_size))
    
    accuracy_a = np.mean(accuracy)
    
    if total_imgs % batch_size != 0:
        batch_input = test_datas[-(total_imgs % batch_size) :]
        batch_labels = test_labels[-(total_imgs % batch_size) :]
        with torch.no_grad():
            batch_output = net(batch_input)
            this_accuracy = calculate_accuracy(batch_output, batch_labels, total_imgs % batch_size)
            
            tested_imgs = total_imgs - total_imgs % batch_size
            remaining_imgs = total_imgs % batch_size
            accuracy_a = (accuracy_a * tested_imgs + this_accuracy * remaining_imgs) / total_imgs
    
    net.train()
    return accuracy_a

def combine_state_dict(dag, state_dict):
    for node in dag.all_nodes:
        if node in dag.input_nodes:
            continue
        
        conv_key = 'convs.' + str(node).replace('.','-') + '.weight'
        conv_bias = 'convs.' + str(node).replace('.','-') + '.bias'
        conv_add_key = 'convs_add.' + str(node).replace('.','-') + '_additional' + '.weight'
        conv_add_bias = 'convs_add.' + str(node).replace('.','-') + '_additional' + '.bias'
        #处理仅有一类链接的情况
        if conv_key not in state_dict.keys():
            if conv_add_key not in state_dict.keys():
                continue
            state_dict[conv_key] = state_dict[conv_add_key]
            del state_dict[conv_add_key]
            if conv_add_bias in state_dict.keys():
                state_dict[conv_bias] = state_dict[conv_add_bias]
                del state_dict[conv_add_bias]
            continue
        if conv_add_key not in state_dict.keys():
            continue
        
        combined_weight = torch.cat((state_dict[conv_key], state_dict[conv_add_key]), dim=1)
        state_dict[conv_key] = combined_weight
        del state_dict[conv_add_key]
        if conv_bias in state_dict.keys():
            state_dict[conv_bias] = state_dict[conv_bias] + state_dict[conv_add_bias]
            del state_dict[conv_add_bias]
    return state_dict

def just_test(savepath):
    nodes_num = 512
    adjoin = 16
    p = 0.6
    
    new_dag = DAG(nodes_num, adjoin, p, "test_model")
 
    new_dag.save_structure(savepath)
    
    loaded_dag = DAG(nodes_num, adjoin, p, "second_model")
    
    loaded_dag.load_structure(savepath, "test_model")
    
    print("********************NODES********************")
    print(loaded_dag.nodes_connections.keys())
    for key, value in new_dag.nodes_connections.items():
        print(key + ":" + str(value))
    print("********************INPUT*******************")
    print(loaded_dag.input_nodes)
    print("********************OUTPUT*******************")
    print(loaded_dag.output_nodes)
    print("********************pooling_gate*******************")
    print(loaded_dag.pooling_gate)
    
    net = Net(loaded_dag, 10, 1)
    net = net.cuda()
    #print(net.convs)
    print(net)
    
    params = list(net.parameters())
    print('Total layers: ' + str(len(params)))
    
    input = torch.randn(200, 3, 32, 32)
    input = input.cuda()
    target = torch.Tensor([0 for i in range(200)]).long()
    target = target.cuda()
    
    START_TIME = time.time()
    output = net(input)
    costs = time.time() - START_TIME
    print(output)
    print("time costs: {0}".format(costs))

if __name__ == '__main__':
    nodes_num = 512
    adjoin = 16
    p = 0.6
    savepath = "./models/"
    DAGname = "WS"+ "_N" + str(nodes_num) + "_K" + str(adjoin) + "_p" + str(p)
    
    new_dag = DAG(nodes_num, adjoin, p, DAGname)
    time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
    model_name = time_now + '_' + DAGname
    new_dag.name = model_name
    new_dag.save_structure(savepath)
    #new_dag.load_structure(savepath, DAGname)
    
    print("********************INPUT*******************")
    print(new_dag.input_nodes)
    print("********************OUTPUT*******************")
    print(new_dag.output_nodes)
    print("********************pooling_gate*******************")
    print(new_dag.pooling_gate)
    
    Learning_Rate =0.1
    Weight_Decay = 1e-4
    Momentum = 0.9
    BatchSize = 64
    dropout = 0
    epoches = 9
    scheduler_step = 3
    BatchSize_test = 1000
    print_frequency = 50
    classes = 10
    
    net = Net(new_dag, classes+1, 
              scale = 4, 
              ext_ratio = 1,
              dropout_rate=dropout, 
              learning_rate = Learning_Rate,
              mome = Momentum, 
              w_d = Weight_Decay,
              scheduler_step = scheduler_step)
    net.apply(weights_init)
    #print(net)
    #net = net.cuda()
    
    print("model ", model_name, " is structured")
    
    torch.save(net.state_dict(), savepath + model_name + '_parameters' + '.pkl')
    print("model saved at", savepath + model_name + '_parameters' + '.pkl')
    
    DATA_DIR = 'lib/CIFAR10_dataset'
    VAL_RATE = 0.1
    Iters = int(50000 * (1-VAL_RATE) // BatchSize)
    
    cifar10 = CIFAR10_Dataset(data_dir=DATA_DIR,
                              val_rate=VAL_RATE)
    
    train_model(net, Iters, cifar10)
    test_model(net, cifar10, BatchSize_test)
