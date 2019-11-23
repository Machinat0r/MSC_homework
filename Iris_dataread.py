# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:24:05 2019

@author: fwd
"""

import numpy as np
import scipy.io as scio

data = scio.loadmat('D:/Anaconda/MSC_homework/Iris/Iris_data.mat')
label = scio.loadmat('D:/Anaconda/MSC_homework/Iris/Iris_label.mat')
data = data['iris1']
label = label['iris1']
print(label)

da = []
la = []
for i in range(150):
    da.append(data[i][0:4])
    la.append(label[i][0])
data = np.mat(da)
label = np.transpose(np.mat(la))

np.savez('iris.npz',data=data,label=label)
