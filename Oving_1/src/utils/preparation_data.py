import os
import numpy as np
from mnist_zip import mnist_basics as mn

__data_files_path__ = os.getcwd() + "/../data_files/"
__mnist_files_path__ = os.getcwd() + "/../mnist_zip/"


def normalize(data: np.ndarray, index: int):
    column = data[:, index]

    avg = column.mean()
    std = column.std()

    data[:, index] = (column - avg) / std


def preparation_normalize(data_list):
    data_column = []
    data_set_size = len(data_list)
    len_targets = len(data_list[0][0])
    for inp, tar in data_list:
        data_column.extend(inp)
    input_matrix = np.array(data_column).reshape(data_set_size, len_targets)
    for index in range(len_targets):
        normalize(input_matrix, index)
    for i in range(len(data_list)):
        data_list[i][0] = list(input_matrix[i])
    data_list = [tuple(e) for e in data_list]
    return data_list


def wine():
    path = __data_files_path__
    data = open(path + "winequality_red.txt", "r")
    data_list = []
    for line in data:
        x = line.split(";")
        inputs = list(map(lambda el: float(el), x[:-1]))
        target = [1 if int(x[-1]) == i else 0 for i in range(3, 9)]
        data_list.append([inputs, target])

    return preparation_normalize(data_list)


def glass():
    path = __data_files_path__
    data = open(path + "glass.txt", "r")
    data_list = []
    for line in data:
        x = line.split(",")
        inputs = list(map(lambda el: float(el), x[:-1]))
        target = [1 if int(x[-1]) == i else 0 for i in range(1, 8)]
        target.pop(3)
        data_list.append([inputs, target])

    return preparation_normalize(data_list)


def yeast():
    path = __data_files_path__
    data = open(path + "yeast.txt", "r")
    data_list = []
    for line in data:
        x = line.split(",")
        inputs = list(map(lambda el: float(el), x[:-1]))
        target = [1 if int(x[-1]) == i else 0 for i in range(1, 10)]

        data_list.append([inputs, target])

    return preparation_normalize(data_list)


def iris():
    path = __data_files_path__
    data = open(path + "iris.txt", "r")
    data_list = []
    for line in data:
        x = line.split(",")
        inputs = list(map(lambda el: float(el), x[:-1]))
        target_name = str(x[-1])

        if target_name.strip() == "Iris-setosa":
            target_num = 0
        elif target_name.strip() == "Iris-versicolor":
            target_num = 1
        else:
            target_num = 2

        target = [1 if target_num == i else 0 for i in range(0, 3)]

        data_list.append([inputs, target])

    return preparation_normalize(data_list)


def mnist(count):
    data_set = mn.load_all_flat_cases(
        dir=__mnist_files_path__)[0:count]

    for i in range(0, len(data_set)):
        data_set[i] = (np.divide(np.array(data_set[i][0]), 256), np.array(data_set[i][1]))

    return data_set
