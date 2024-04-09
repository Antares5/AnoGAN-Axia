import numpy as np
import pandas as pd
import os

import torch
from skimage import color, transform, io


def train_images_load(folder_path):
    # 获取给定文件夹内所有图片文件的路径
    img_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                 os.path.isfile(os.path.join(folder_path, f))]

    images = []
    for file in img_files:
        # 读取图片并转换为灰度图
        img = io.imread(file)

        subimages = image_split(img, subimage_size=(64, 64))
        images.extend(subimages)

    return np.array(images)


def test_images_load(folder_path):
    first_images = []
    other_images = []

    priority_path = os.path.join(folder_path, 'good')
    priority_files = [os.path.join(priority_path, f) for f in os.listdir(priority_path) if
                      os.path.isfile(os.path.join(priority_path, f))]
    for file in priority_files:
        img = io.imread(file)
        subfirst = image_split(img)
        first_images.extend(subfirst)

    for subdir, dirs, files in os.walk(folder_path):
        if subdir == priority_path:
            continue  # 已处理过优先目录，跳过
        for file in files:
            img_path = os.path.join(subdir, file)
            img = io.imread(img_path)
            subother = image_split(img)
            other_images.extend(subother)

    images = first_images + other_images
    return np.array(images)


def image_split(image, subimage_size=(64, 64)):
    subimages = []
    for i in range(0, image.shape[0], subimage_size[0]):
        for j in range(0, image.shape[1], subimage_size[1]):
            subimage = image[i: i + subimage_size[0], j: j + subimage_size[1]]
            if subimage.shape[0] == subimage_size[0] and subimage.shape[1] == subimage_size[1]:
                subimages.append(subimage)
    return subimages


# 用你的图片文件夹路径替换这里
train_path = './/data//MVTecAD//grid//train//good'
test_path = r'.\data\MVTecAD\grid\test'

# 调用函数
loaded_train_set = train_images_load(train_path)
loaded_test_set = test_images_load(test_path)

# 输出结果以确认加载成功
print("Loaded train images shape:", loaded_train_set.shape)
print("Loaded test images shape:", loaded_test_set.shape)

# MVTecAD_grid_train_resized = np.array([transform.resize(img, (64, 64)).astype(np.float32) for img in loaded_train_set])
# MVTecAD_grid_test_resized = np.array([transform.resize(img, (64, 64)).astype(np.float32) for img in loaded_test_set])
MVTecAD_grid_train_resized = loaded_train_set
MVTecAD_grid_test_resized = loaded_test_set
print("Train set shape:", MVTecAD_grid_train_resized.shape)
print("Test set shape:", MVTecAD_grid_test_resized.shape)
