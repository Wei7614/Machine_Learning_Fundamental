# pocket pla

import numpy as np
import os
import math

training_data_source = np.loadtxt('./hw1_18_train.dat.txt')
testing_data_source = np.loadtxt('./hw1_18_test.dat.txt')


def sign(z):
    if z > 0:
        return 1
    else:
        return -1


def get_error_sum(w, data_set, target_set):
    error_sum = 0
    for idx in range(len(data_set)):
        x = np.concatenate((np.array([1, ]), data_set[idx, :]))
        y = target_set[idx]
        if sign(np.dot(w, x)) != y:
            error_sum += 1
    return error_sum


def get_x_y(dataset, targetset, idx):
    x = np.concatenate((np.array([1, ]), dataset[idx, :]))
    y = targetset[idx]
    return x, y


error_sum = float(math.inf)
error_total_train = 0
error_total_test = 0
# 實驗兩千次
for exp_times in range(0, 2000):

    # 取前面100個做實驗
    data_length = 100
    training_data = training_data_source[:data_length, 0:4]
    training_target = training_data_source[:data_length, 4]

    testing_data = testing_data_source[:, 0:4]
    testing_target = testing_data_source[:, 4]

    w = np.random.rand(1, training_data.shape[1] + 1)
    for idx in range(len(training_data)):
        x, y = get_x_y(training_data, training_target, idx)
        if sign(np.dot(w, x)) != y:
            w_new = w + y * x
            # 如果表現的比較好再更換
            if get_error_sum(w_new, training_data, training_target) < error_sum:
                w = w_new

    error_train = 0
    for idx in range(len(training_data)):
        x, y = get_x_y(training_data, training_target, idx)
        if sign(np.dot(w, x)) != y:
            error_train += 1
    error_total_train += error_train/len(training_data)

    # 將w去test_set做評估
    error_test = 0
    for idx in range(len(testing_data)):
        x, y = get_x_y(testing_data, testing_target, idx)
        if sign(np.dot(w, x)) != y:
            error_test += 1
    error_total_test += error_test/len(testing_data)

# error_train 約 0.23, error_test 約 0.27 for 50 updates
# error_train 約 0.26, error_test 約 0.25 for 100 updates
print("error_train = {}, error_test = {}".format(error_total_train/2000, error_total_test/2000))



