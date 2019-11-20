# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 00:07:24 2019

@author: fwd
"""

from sklearn.datasets import *
import numpy as np
boston = load_boston().data
data = []
target = []
for i in range(506):
    data.append(boston[i][0:12])
    target.append(boston[i][12])
    
data = np.mat(data)
target = np.transpose(np.mat(target))

np.savez('boston.npz',data=data,label=target)