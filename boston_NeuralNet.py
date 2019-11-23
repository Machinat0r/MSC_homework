# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 00:19:46 2019

@author: fwd
"""

import numpy as np
#from sklearn.datasets import load_boston
from HelperClass.NeuralNet_1_1 import *
from level4_DeNormalizeWB import *
from level3_DataNormalization import *

file_name = "D:/Anaconda/boston.npz"

if __name__ == '__main__':
    # data
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    reader.NormalizeX()
    reader.NormalizeY()
    # net
    hp = HyperParameters_1_0(12,1,eta=0.001, max_epoch=10000, batch_size=50, eps=0.01)
    net = NeuralNet_1_1(hp)
    net.train(reader, checkpoint=0.1)
    # inference
    W_real, B_real = DeNormalizeWeightsBias(net, reader)
    print("W_real=", W_real)
    print("B_real=", B_real)
    
    x = np.array([0.10328,25,5.13,0,0.453,5.927,47.2,6.932,8,284,19.7,396.9]).reshape(1,12)
    x_new = reader.NormalizePredicateData(x)
#    z = np.dot(x,W_real) + B_real
    z = net.inference(x_new)
    print('z=', z)
    z_real = z*reader.Y_norm[0,1] + reader.Y_norm[0,0]
    print('z_real=', z_real)

#    ShowResult(net,reader)