# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:15:00 2019

@author: fwd
"""
#神经网络训练完成后会停下，此时输入类似4.9,3.1,1.5,0.2这样的数组就会返回分类结果
import numpy as np

from HelperClass.NeuralNet_1_2 import *

file_name = "D:/Anaconda/MSC_homework/Iris/iris.npz"

def inference(net, reader):
    xt_raw = np.array([a1,a2,a3,a4]).reshape(1,4)
    xt = reader.NormalizePredicateData(xt_raw)
    output = net.inference(xt)
    r = np.argmax(output, axis=1)+1
    cat = {1:'Iris Setosa',2:'Iris Versicolour',3:'Iris Virginica'}
    r1 = cat[int(r)]
    print("output=", output)
    print("类别：", r1)

# 主程序
if __name__ == '__main__':
    num_category = 3
    reader = DataReader_1_3(file_name)
    reader.ReadData()
    reader.NormalizeX()
    reader.ToOneHot(num_category, base=1)

    num_input = 4
    params = HyperParameters_1_1(num_input, num_category, eta=0.1, max_epoch=1000, batch_size=10, eps=1e-3, net_type=NetType.MultipleClassifier)
    net = NeuralNet_1_2(params)
    net.train(reader, checkpoint=1)
    
    a1,a2,a3,a4 = map(float,input().split(','))
    inference(net, reader)