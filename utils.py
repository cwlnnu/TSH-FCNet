# -*- coding:utf-8 -*-

import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
import numpy as np
import h5py

def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device

def seed_worker(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.


def get_dataset(dataset_name):
    if dataset_name == 'Houston':
        DataPath1 = './data/Houston/Houston_HS.mat'
        DataPath2 = './data/Houston/Houston_MS.mat'
        DataPath3 = './data/Houston/Houston_LiDAR.mat'
        Label_train = './data/Houston/Houston_train.mat'
        Label_test = './data/Houston/Houston_test.mat'
        Data1 = loadmat(DataPath1)['Houston_HS']
        Data2 = loadmat(DataPath2)['Houston_MS']
        Data3 = loadmat(DataPath3)['Houston_LiDAR']
        gt_train = loadmat(Label_train)['Houston_train']
        gt_test = loadmat(Label_test)['Houston_test']

    elif dataset_name == 'Augsburg_City':
        DataPath1 = './data/Augsburg_City/Augsburg_City_HS.mat'
        DataPath2 = './data/Augsburg_City/Augsburg_City_MS.mat'
        DataPath3 = './data/Augsburg_City/Augsburg_City_SAR.mat'
        Label_train = './data/Augsburg_City/Augsburg_City_train_200.mat'
        Label_test = './data/Augsburg_City/Augsburg_City_test_200.mat'
        Data1 = h5py.File(DataPath1)['Augsburg_City_HS']
        Data2 = loadmat(DataPath2)['Augsburg_City_MS']
        Data3 = loadmat(DataPath3)['Augsburg_City_SAR']
        gt_train = loadmat(Label_train)['Augsburg_City_train']
        gt_test = loadmat(Label_test)['Augsburg_City_test']

    elif dataset_name == 'Beijing':
        DataPath1 = './data/Beijing/Beijing_HS.mat'
        DataPath2 = './data/Beijing/Beijing_MS.mat'
        DataPath3 = './data/Beijing/Beijing_SAR.mat'
        Label_train = './data/Beijing/Beijing_train.mat'
        Label_test = './data/Beijing/Beijing_test.mat'
        Data1 = loadmat(DataPath1)['Beijing_HS']
        Data2 = loadmat(DataPath2)['Beijing_MS']
        Data3 = loadmat(DataPath3)['Beijing_SAR']
        gt_train = loadmat(Label_train)['Beijing_train']
        gt_test = loadmat(Label_test)['Beijing_test']

    Data1 = (Data1 - np.min(Data1)) / (np.max(Data1) - np.min(Data1))
    Data2 = (Data2 - np.min(Data2)) / (np.max(Data2) - np.min(Data2))
    Data3 = (Data3 - np.min(Data3)) / (np.max(Data3) - np.min(Data3))

    return Data1, Data2, Data3, gt_train, gt_test


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


def sampling(groundTruth):
    select = {}
    m = int(groundTruth.max())

    for i in range(m):
        indices = [
            j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1
        ]
        np.random.shuffle(indices)
        select[i] = indices[:]

    select_indices = []

    for i in range(m):
        select_indices += select[i]
    np.random.shuffle(select_indices)
    return select_indices

def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA, matrix


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))





