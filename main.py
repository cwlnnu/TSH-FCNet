# -*- coding:utf-8 -*-

import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from TriFusion_Net import TriFusion
import numpy as np
import time
from utils import AverageMeter, accuracy, output_metric, print_args, get_device, seed_worker, get_dataset, sampling, select_small_cubic
import torch

# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser("TriFusion")
parser.add_argument('--seed', type=int, default=40, help='number of seed')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--num_epoch', type=int, default=500, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--cuda', type=int, default=0, help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--dataset', choices=['Augsburg_City', 'Beijing', 'Houston'], default='Houston', help='dataset to use')
parser.add_argument('--num_classes', choices=[13, 13, 15], default=15, help='number of classes')
parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--patch_size', type=int, default=9, help='number of patch size')

args = parser.parse_args()
DEVICE = get_device(args.cuda)
softmax = nn.Softmax(dim=1)

def main():
    seed_worker(args.seed)

    Data1, Data2, Data3, gt_train, gt_test = get_dataset(args.dataset)

    if Data1.shape[0] != Data2.shape[0]:
        Data1 = Data1.transpose(2, 1, 0)

    if Data1.ndim == 2:
        Data1 = Data1[..., np.newaxis]
    elif Data2.ndim == 2:
        Data2 = Data2[..., np.newaxis]
    elif Data3.ndim == 2:
        Data3 = Data3[..., np.newaxis]

    Data1 = Data1.astype(np.float32)
    Data2 = Data2.astype(np.float32)
    Data3 = Data3.astype(np.float32)

    INPUT_DIMENSION_1 = Data1.shape[2]
    INPUT_DIMENSION_2 = Data2.shape[2]
    INPUT_DIMENSION_3 = Data3.shape[2]

    num_classes = args.num_classes
    r = int(args.patch_size / 2)

    padded_data_1 = np.lib.pad(Data1, ((r, r), (r, r), (0, 0)), 'constant', constant_values=0)
    padded_data_2 = np.lib.pad(Data2, ((r, r), (r, r), (0, 0)), 'constant', constant_values=0)
    padded_data_3 = np.lib.pad(Data3, ((r, r), (r, r), (0, 0)), 'constant', constant_values=0)

    train_indices = sampling(gt_train)
    Test_indices = sampling(gt_test)

    train_gt = gt_train.reshape(np.prod(gt_train.shape[:2]), )
    test_gt = gt_test.reshape(np.prod(gt_test.shape[:2]), )

    TRAIN_SIZE = len(train_indices)
    TEST_SIZE = len(Test_indices)

    y_train = train_gt[train_indices] - 1
    y_test = test_gt[Test_indices] - 1

    train_data_1 = select_small_cubic(TRAIN_SIZE, train_indices, Data1, r, padded_data_1, INPUT_DIMENSION_1)
    train_data_2 = select_small_cubic(TRAIN_SIZE, train_indices, Data2, r, padded_data_2, INPUT_DIMENSION_2)
    train_data_3 = select_small_cubic(TRAIN_SIZE, train_indices, Data3, r, padded_data_3, INPUT_DIMENSION_3)

    test_data_1 = select_small_cubic(TEST_SIZE, Test_indices, Data1, r, padded_data_1, INPUT_DIMENSION_1)
    test_data_2 = select_small_cubic(TEST_SIZE, Test_indices, Data2, r, padded_data_2, INPUT_DIMENSION_2)
    test_data_3 = select_small_cubic(TEST_SIZE, Test_indices, Data3, r, padded_data_3, INPUT_DIMENSION_3)

    train_data_1_tensor = torch.from_numpy(train_data_1).type(torch.FloatTensor)
    train_data_2_tensor = torch.from_numpy(train_data_2).type(torch.FloatTensor)
    train_data_3_tensor = torch.from_numpy(train_data_3).type(torch.FloatTensor)
    train_label_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)

    test_data_1_tensor = torch.from_numpy(test_data_1).type(torch.FloatTensor)
    test_data_2_tensor = torch.from_numpy(test_data_2).type(torch.FloatTensor)
    test_data_3_tensor = torch.from_numpy(test_data_3).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(y_test).type(torch.FloatTensor)

    torch_dataset_train = Data.TensorDataset(train_data_1_tensor, train_data_2_tensor, train_data_3_tensor, train_label_tensor)
    torch_dataset_test = Data.TensorDataset(test_data_1_tensor, test_data_2_tensor, test_data_3_tensor, test_label_tensor)

    train_loader = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    len_train_dataset = len(train_loader.dataset)
    len_test_dataset = len(test_loader.dataset)

    print("train samples :", len_train_dataset)
    print("test samples:", len_test_dataset)

    model = TriFusion(INPUT_DIMENSION_1, INPUT_DIMENSION_2, INPUT_DIMENSION_3, num_classes)
    model = model.to(DEVICE)

    loss_cal = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.num_epoch // 10, gamma=args.gamma)

    BestAcc = 0
    val_acc = []
    t1 = time.time()

    for epoch in range(1, args.num_epoch + 1):
        model.train()
        train_acc, train_obj, tar_t, pre_t = train(model, train_loader, loss_cal, optimizer)

        OA1, AA1, Kappa1, CA1, matrix1 = output_metric(tar_t, pre_t)
        print("Epoch: {:03d} | train_loss: {:.4f} | train_OA: {:.4f} | train_AA: {:.4f} | train_Kappa: {:.4f}"
              .format(epoch, train_obj, OA1, AA1, Kappa1))
        scheduler.step()

        if (epoch % args.test_freq == 0) | (epoch == args.num_epoch):
            model.eval()
            tar_v, pre_v = test(model, test_loader, loss_cal)
            OA2, AA2, Kappa2, CA2, matrix2 = output_metric(tar_v, pre_v)
            val_acc.append(OA2)
            print("Every 5 epochs' records:")
            print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA2, Kappa2))
            print(CA2)
            if OA2 > BestAcc:
                torch.save(model.state_dict(), './TriFusion_Net.pkl')
                BestAcc = OA2

    t2 = time.time()
    model.eval()
    model.load_state_dict(torch.load('./TriFusion_Net.pkl'))
    tar_v, pre_v = test(model, test_loader, loss_cal)
    OA, AA, Kappa, CA, matrix = output_metric(tar_v, pre_v)
    print("Final records:")
    print("Maxmial Accuracy: %f, index: %i" % (max(val_acc), val_acc.index(max(val_acc))))
    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa))
    print(CA)
    print(matrix)
    print("Running Time: {:.2f}".format(t2 - t1))
    print("**************************************************")
    print("Parameter:")
    print_args(vars(args))


def train(model, train_loader, loss_cal, optimizer):
    objs = AverageMeter()
    top1 = AverageMeter()
    tar = np.array([])
    pre = np.array([])

    for X1, X2, X3, y in train_loader:
        X1 = X1.to(DEVICE)
        X2 = X2.to(DEVICE)
        X3 = X3.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()

        y_pred_1, wd, gwd = model(X1, X2, X3)
        loss_1 = loss_cal(y_pred_1, y.long())
        loss_got = wd * 1 + gwd * 1
        loss = loss_1 + loss_got

        loss.backward()
        optimizer.step()

        y_pred_1 = softmax(y_pred_1)

        prec1, t, p = accuracy(y_pred_1, y, topk=(1,))
        n = X1.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre


def test(model, test_loader, loss_cal):
    objs = AverageMeter()
    top1 = AverageMeter()
    tar = np.array([])
    pre = np.array([])

    for X1, X2, X3, y in test_loader:
        X1 = X1.to(DEVICE)
        X2 = X2.to(DEVICE)
        X3 = X3.to(DEVICE)
        y = y.to(DEVICE)

        y_pred_1, wd, gwd = model(X1, X2, X3)
        loss_1 = loss_cal(y_pred_1, y.long())
        loss_got = wd * 1 + gwd * 1
        loss = loss_1 + loss_got

        y_pred_1 = softmax(y_pred_1)

        prec1, t, p = accuracy(y_pred_1, y, topk=(1,))
        n = X1.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre


if __name__ == '__main__':
    main()




