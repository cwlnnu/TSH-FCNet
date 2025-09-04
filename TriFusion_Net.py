# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from OT_torch_ import cost_matrix_batch_torch, GW_distance_uniform, IPOT_distance_torch_batch_uniform


def OT(source_share, target_share, beta, ori=False):
    if ori == True:
        source_share = source_share.x.unsqueeze(0).transpose(2,1)
        target_share = target_share.x.unsqueeze(0).transpose(2,1)
    else:
        source_share = source_share.unsqueeze(0).transpose(2,1)
        target_share = target_share.unsqueeze(0).transpose(2,1)

    cos_distance = cost_matrix_batch_torch(source_share, target_share)
    cos_distance = cos_distance.transpose(1,2)
    # TODO: GW and Gwd as graph alignment loss
    min_score = cos_distance.min()
    max_score = cos_distance.max()
    threshold = min_score + beta * (max_score - min_score)
    cos_dist = torch.nn.functional.relu(cos_distance - threshold)
    wd = - IPOT_distance_torch_batch_uniform(cos_dist, source_share.size(0), source_share.size(2), target_share.size(2), iteration=30)
    gwd = GW_distance_uniform(source_share, target_share, beta)
    return torch.mean(wd), torch.mean(gwd)


def detect_edge(x):
    b, c, h, w = x.size()
    kernal_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).expand(c, -1, -1).view(1, c, 3, 3)
    kernal_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).expand(c, -1, -1).view(1, c, 3, 3)
    edge_x = F.conv2d(x, kernal_x.cuda(), padding=1)
    edge_y = F.conv2d(x, kernal_y.cuda(), padding=1)

    edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)

    return edge_magnitude


def cal_similarity(x, sigma=0.8):
    _, c, h, w = x.shape
    x_1 = torch.unsqueeze(x.view(-1, c, h * w), 3)
    x_2 = torch.unsqueeze(x.view(-1, c, h * w), 2)
    distance = torch.norm(x_1 - x_2, dim=1)
    similarity = torch.exp(-distance ** 2 / (2 * (sigma ** 2)))

    return similarity


class Feature_Align_S(nn.Module):
    def __init__(self, channel):
        super(Feature_Align_S, self).__init__()

        self.value_conv = nn.Conv2d(in_channels=channel, out_channels=channel // 8, kernel_size=1)
        self.conv = nn.Conv2d(in_channels=channel // 8, out_channels=channel, kernel_size=1)
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.gamma2 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.gamma3 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.gamma4 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.gamma5 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.gamma6 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.gamma7 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.gamma8 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.gamma9 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self, data1, data2, data3):
        b, c, h, w = data1.size()

        attention_s1 = cal_similarity(data1, self.gamma1)
        attention_s2 = cal_similarity(data2, self.gamma2)
        attention_s3 = cal_similarity(data3, self.gamma3)

        value1 = self.value_conv(data1).view(b, -1, h * w).permute(0, 2, 1)
        value2 = self.value_conv(data2).view(b, -1, h * w).permute(0, 2, 1)
        value3 = self.value_conv(data3).view(b, -1, h * w).permute(0, 2, 1)
        attention = attention_s1 * self.gamma7 + attention_s2 * self.gamma8 + attention_s3 * self.gamma9

        att_s11 = torch.bmm(attention, value1).permute(0, 2, 1).view(b, -1, h, w)
        att_s12 = torch.bmm(attention_s1, value2).permute(0, 2, 1).view(b, -1, h, w)
        att_s13 = torch.bmm(attention_s1, value3).permute(0, 2, 1).view(b, -1, h, w)

        att_s21 = torch.bmm(attention_s2, value1).permute(0, 2, 1).view(b, -1, h, w)
        att_s22 = torch.bmm(attention, value2).permute(0, 2, 1).view(b, -1, h, w)
        att_s23 = torch.bmm(attention_s2, value3).permute(0, 2, 1).view(b, -1, h, w)

        att_s31 = torch.bmm(attention_s3, value1).permute(0, 2, 1).view(b, -1, h, w)
        att_s32 = torch.bmm(attention_s3, value2).permute(0, 2, 1).view(b, -1, h, w)
        att_s33 = torch.bmm(attention, value3).permute(0, 2, 1).view(b, -1, h, w)

        att_1 = (att_s11 + att_s21 + att_s31) * self.gamma4
        att_2 = (att_s22 + att_s12 + att_s32) * self.gamma5
        att_3 = (att_s33 + att_s13 + att_s23) * self.gamma6
        att_1 = self.conv(att_1)
        att_2 = self.conv(att_2)
        att_3 = self.conv(att_3)

        data11 = data1 * att_1
        data22 = data2 * att_2
        data33 = data3 * att_3

        return data11, data22, data33


class CNN_Encoder(nn.Module):
    def __init__(self, band1, band2, band3):
        super(CNN_Encoder, self).__init__()
        self.align_s = Feature_Align_S(64)
        self.gamma1 = torch.nn.Parameter(torch.tensor(0.2))

        self.conv11 = nn.Sequential(
            nn.Conv2d(band1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
        )

        self.conv21 = nn.Sequential(
            nn.Conv2d(band2, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
        )
        self.conv23 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
        )

        self.conv31 = nn.Sequential(
            nn.Conv2d(band3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
        )
        self.conv32 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
        )
        self.conv33 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
        )
        self.common1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
        )
        self.common2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x1, x2, x3):
        edge3 = detect_edge(x3)

        x1 = self.conv11(x1)
        x2 = self.conv21(x2)
        x3 = self.conv31(x3)
        x11, x22, x33 = x1, x2, x3

        x1 = self.conv12(x1)
        x2 = self.conv22(x2)
        x3 = self.conv32(x3)

        x1 = self.conv13(x1)
        x2 = self.conv23(x2)
        x3 = self.conv33(x3)

        x1_common = self.common1(x11)
        x1_common = self.common2(x1_common)

        x2_common = self.common1(x22)
        x2_common = self.common2(x2_common)

        x3_common = self.common1(x33)
        x3_common = self.common2(x3_common)

        add_coarse = x1_common + x2_common + x3_common
        add = x1 + x2 + x3

        add_1 = add.reshape(add.size(0), -1)
        add_2 = add_coarse.reshape(add_coarse.size(0), -1)

        wd, gwd = OT(add_1, add_2, self.gamma1)

        xs1, xs2, xs3 = self.align_s(x1, x2, x3)
        x1 = xs1 + edge3
        x2 = xs2 + edge3
        x3 = xs3 + edge3

        return x1, x2, x3, wd, gwd


class CNN_Classifier(nn.Module):
    def __init__(self, Classes):
        super(CNN_Classifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, Classes, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x


class TriFusion(nn.Module):
    def __init__(self, band1, band2, band3, num_classes):
        super().__init__()
        self.cnn_encoder = CNN_Encoder(band1, band2, band3)

        self.cnn_classifier = CNN_Classifier(num_classes)

    def forward(self, data1, data2, data3):
        data1 = data1.permute(0, 3, 1, 2)
        data2 = data2.permute(0, 3, 1, 2)
        data3 = data3.permute(0, 3, 1, 2)

        out1, out2, out3, wd, gwd = self.cnn_encoder(data1, data2, data3)

        add = out1 + out2 + out3

        pred_1 = self.cnn_classifier(add)

        return pred_1, wd, gwd
