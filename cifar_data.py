import numpy as np
import pandas as pd

import torch
from skimage import color, transform
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

from tensorflow.keras.datasets import cifar10

'''
Cifar数据读取
'''

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 查询训练数据中标签为某些类别的数据，并取前一定数量的样本
selected_train_classes = [9]
selected_test_classes = [8,9]
num_samples_train = 5000
num_samples_test = 2000

train_indices = np.isin(y_train.flatten(), selected_train_classes)
test_indices_pos = np.isin(y_test.flatten(), selected_test_classes)

x_train_selected = x_train[train_indices][:num_samples_train]
x_test_selected = x_test[test_indices_pos][:num_samples_test]

# 转换为灰度图像，并将数据类型转换为float32
x_train_gray = np.array([color.rgb2gray(img) for img in x_train_selected])
x_test_gray = np.array([color.rgb2gray(img) for img in x_test_selected])
x_train_resized = np.array([transform.resize(img, (64, 64)).astype(np.float32) for img in x_train_gray])
x_test_resized = np.array([transform.resize(img, (64, 64)).astype(np.float32) for img in x_test_gray])

# 显示转换后的图像大小
print(x_train_resized.shape)
print(x_test_resized.shape)