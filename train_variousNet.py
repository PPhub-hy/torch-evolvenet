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
    
    New_dag = True
    evolving_round = 3

    nodes_num = 512
    edges = 8192
    SCALE = 4
    
    CUTOUT = False
    
    if New_dag:
        in_nodes = 3
        out_nodes = 11
        
        savepath = "./models/"
        DAGname = "rand_N" + str(nodes_num) + "_edge" + str(edges) + "_scale" + str(SCALE) + "_co" + str(CUTOUT)
    
        #新建DAG，存储模型结构
        dag = DAG(nodes_num, edges, in_nodes, out_nodes, name = DAGname)#, rolling_rate
        time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
        model_name = time_now + '_' + DAGname
        os.mkdir(savepath + model_name + '/')
        savepath = savepath + model_name + '/'
        
        dag.name = model_name
        dag.save_structure(savepath)
    
    else:
        #装载原DAG
        dag_loading_path = "./models/"
        dag_loading_name = '2019-12-23_11:13:32_rand_N512_edge8192_InOut3-11'
        if evolving_round == 2:
            dag_loading_path = dag_loading_path + dag_loading_name + '/iter20/'
        else:
            dag_loading_path = dag_loading_path + dag_loading_name + '_round' + str(evolving_round - 1) + '/iter20/'
        dag = DAG(empty = True)
        dag.load_structure(dag_loading_path, dag_loading_name)
        savepath = "./models/" + dag_loading_name + '_round' + str(evolving_round) + '/'
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
    Ext_ratio = 1
    Leaky_a = 0.1
    Reslink = 'None' # 'None'、'flow'、'stream'
    multi_gpu = False
    epoches_consine = 500
    epoches = 40
    last_epoch = -1
    scheduler_step = [80, 140, 170]
    
    #初始化模型
    print('cosine from {} to {} in {} epoches'.format(Learning_Rate, lr_min, epoches_consine))
    print('here start training at {}th iter'.format(last_epoch))
    net = torch_EvolveNet.Net(dag, classes+1, 
                              scale = SCALE, 
                              ext_ratio = Ext_ratio,
                              dropout_rate=dropout, 
                              learning_rate = Learning_Rate,
                              lr_min = lr_min,
                              last_epoch = last_epoch,
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
    last_epoch += epoches
    
    #保存模型参数文件
    torch.save(net, savepath + net.network.name + '_model.pkl')
    print("model saved at", savepath + net.network.name + '_model.pkl')
    
    if Evolving:
        print("start model iteration!")
        tick_times = 10
        REDUCE_RATIO_init = 0.01
        REDUCE_RATIO_final = 0.001
        CONCENTRATION = 2
        ALL_iteration = 21
    
        for current_iteration in range(ALL_iteration):
            if multi_gpu:
                last_lr = net.module.scheduler.get_lr()[0]
            else:
                last_lr = net.scheduler.get_lr()[0]
            
            REDUCE_RATIO = REDUCE_RATIO_final + 0.5 * (REDUCE_RATIO_init - REDUCE_RATIO_final) * (1 + math.cos(current_iteration / ALL_iteration * math.pi))
            FREEZE_RATIO = last_lr / 0.1
            REDUCE_NUM = int(edges * REDUCE_RATIO + (dag.sum_edges_num()[0] - edges) / tick_times)
            EXTEND_NUM = int(edges * REDUCE_RATIO) * tick_times
            print('----------------------------------')
            print('Iteration', current_iteration + 1, 'start!')
            print("parameters:")
            print("tick_times = ", tick_times)
            print("REDUCE_NUM(tick) = ", REDUCE_NUM  * tick_times)
            print("EXTEND_NUM(tick) = ", EXTEND_NUM)
            print("CONCENTRATION = ", CONCENTRATION)
            iter_path = savepath + 'iter' + str(current_iteration) + '/'
            os.mkdir(iter_path)
            
            '''GFIR训练'''
            #训练计划
            epoches = 3
            lr_min = 0.02
            
            
            print('start to train edge masks!')
            #获取原模型参数字典
            if multi_gpu:
                old_dict = net.module.state_dict()
            else:
                old_dict = net.state_dict()
            print('cosine from {} to {} in {} epoches'.format(Learning_Rate, lr_min, epoches))
            print('here start training at {}th iter'.format(-1))
            net = torch_EvolveNet.Net(dag, classes+1, 
                                      scale = SCALE, 
                                      ext_ratio = Ext_ratio,
                                      dropout_rate=dropout, 
                                      learning_rate = Learning_Rate,
                                      lr_min = lr_min,
                                      last_epoch = -1,
                                      freeze_ratio=FREEZE_RATIO,
                                      mome = Momentum, 
                                      w_d = Weight_Decay,
                                      #scheduler_step = scheduler_step,
                                      epoches = epoches,
                                      leaky_a = Leaky_a,
                                      Res = Reslink,
                                      GFIR = True)
            new_state = net.state_dict()
            new_state.update(old_dict)
            net.load_state_dict(new_state)
            print(torch.cuda.device_count(), 'GPU is used')
            
            #GPU模型转移
            if multi_gpu:
                net = nn.DataParallel(net.cuda())
            else:
                net = net.cuda()

            #训练masks
            torch_EvolveNet.train_model(net, Iters, cifar10, train_generator, epoches, scheduler_step, BatchSize, val_BatchSize, print_frequency, logpath=iter_path, multi_gpu = multi_gpu)
            
            #prune
            Need_extend = False
            for idx in range(tick_times):
                """按照 GFIR or 卷积核范数大小 更新结构"""
                print('ticking : {}/{}'.format(idx + 1,tick_times))
                if idx == tick_times - 1:
                    Need_extend = True
                new_state, dag = analysis_connections.DAG_iteration(net = net,
                                                                    multi_gpu = multi_gpu,
                                                                    dag = dag, #model
                                                                    channels = SCALE,
                                                                    savepath = iter_path, # an intermediate to save all link strength of every edge
                                                                    iteration = current_iteration, # The number of iterations in DAG structure
                                                                    reduce_num = REDUCE_NUM, # shows how many edges will be remove
                                                                    extend_num = EXTEND_NUM, # shows how many edges will be relink
                                                                    concentration = CONCENTRATION,
                                                                    dataset = cifar10,
                                                                    val_number = 50000 * VAL_RATE,
                                                                    Need_extend = Need_extend) # concentration of delated edges (bigger has smaller randomness)
                if not idx == tick_times - 1:
                    net = torch_EvolveNet.Net(dag, classes+1, 
                                              scale = SCALE, 
                                              ext_ratio = Ext_ratio,
                                              dropout_rate=dropout, 
                                              learning_rate = Learning_Rate,
                                              lr_min = lr_min,
                                              last_epoch = -1,
                                              freeze_ratio=FREEZE_RATIO,
                                              mome = Momentum, 
                                              w_d = Weight_Decay,
                                              #scheduler_step = scheduler_step,
                                              epoches = epoches,
                                              leaky_a = Leaky_a,
                                              Res = Reslink,
                                              GFIR = True)
                    state_dict = net.state_dict()
                    new_state = {k:v for k,v in new_state.items() if k in state_dict.keys()}
                    net.load_state_dict(new_state)
                    #GPU模型转移
                    if multi_gpu:
                        net = nn.DataParallel(net.cuda())
                    else:
                        net = net.cuda()
            
            '''预热训练'''
            #训练计划
            epoches = 3 + int((current_iteration+1)/3)
            lr_min = last_lr

            #装载新网络
            print('cosine from {} to {} in {} epoches'.format(Learning_Rate, lr_min, epoches))
            print('here start training at {}th iter'.format(-1))
            net = torch_EvolveNet.Net(dag, classes+1, 
                                      scale = SCALE, 
                                      ext_ratio = Ext_ratio,
                                      dropout_rate=dropout, 
                                      learning_rate = Learning_Rate,
                                      lr_min = lr_min,
                                      last_epoch = -1,
                                      freeze_ratio=FREEZE_RATIO,
                                      mome = Momentum, 
                                      w_d = Weight_Decay,
                                      #scheduler_step = scheduler_step,
                                      epoches = epoches,
                                      leaky_a = Leaky_a,
                                      Res = Reslink)
            net.apply(torch_EvolveNet.weights_init)
            #print(net)
            
            #将 new_state 中的新参数在 model_dict 中更新
            state_dict = net.state_dict()
            new_state = {k:v for k,v in new_state.items() if k in state_dict.keys()} #删去mask参数
            state_dict.update(new_state)
            #读取网络参数
            net.load_state_dict(state_dict)
            if multi_gpu:
                net = nn.DataParallel(net.cuda())
            else:
                net = net.cuda()
            
            #训练模型
            torch_EvolveNet.train_model(net, Iters, cifar10, train_generator, epoches, scheduler_step, BatchSize, val_BatchSize, print_frequency, logpath=iter_path, multi_gpu = multi_gpu)
            
            #connection combination of dag
            dag.combine_connections()
            
            #保存新DAG
            dag.save_structure(iter_path)
            
            #connection combination of state_dict
            if multi_gpu:
                state_dict = torch_EvolveNet.combine_state_dict(dag, net.module.state_dict())
            else:
                state_dict = torch_EvolveNet.combine_state_dict(dag, net.state_dict())
            
            '''继续前进'''    
            #训练计划
            if current_iteration + 1 == ALL_iteration:
                epoches = 60
                lr_min = 1e-5
            else:
                epoches = 20
                lr_min = 1e-5
            #装载新网络
            print('cosine from {} to {} in {} epoches'.format(Learning_Rate, lr_min, epoches_consine))
            print('here start training at {}th iter'.format(last_epoch))
            net = torch_EvolveNet.Net(dag, classes+1, 
                                      scale = SCALE, 
                                      ext_ratio = Ext_ratio,
                                      dropout_rate=dropout, 
                                      learning_rate = Learning_Rate,
                                      lr_min = lr_min,
                                      last_epoch = last_epoch,
                                      freeze_ratio=FREEZE_RATIO,
                                      mome = Momentum, 
                                      w_d = Weight_Decay,
                                      #scheduler_step = scheduler_step,
                                      epoches = epoches_consine,
                                      leaky_a = Leaky_a,
                                      Res = Reslink)
            
            net.load_state_dict(state_dict)
            if multi_gpu:
                net = nn.DataParallel(net.cuda())
            else:
                net = net.cuda()
            #训练模型
            torch_EvolveNet.train_model(net, Iters, cifar10, train_generator, epoches, scheduler_step, BatchSize, val_BatchSize, print_frequency, logpath=iter_path, multi_gpu = multi_gpu)
            last_epoch += epoches
            
            #保存模型参数文件
            torch.save(net, iter_path + net.network.name + '_model.pkl')
            print("model saved at", iter_path + net.network.name + '_model.pkl')
            
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
    