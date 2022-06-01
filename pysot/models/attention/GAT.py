import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Graph_Attention_Union, self).__init__()

        # search region nodes linear transformation
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1)

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        # aggregated feature
        self.fi = nn.Sequential(
            nn.Conv2d(in_channel*2, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        # self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, zf, xf):
        # linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        xf_g = self.g(xf)
        zf_g = self.g(zf)


        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)

        similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        similar = F.softmax(similar, dim=2)

        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])



        # aggregated feature
        output = torch.cat([embedding, xf_g], 1)
        output = self.fi(output)
        # output = self.conv1(output)
        # output = self.conv2(output)
        return output
if __name__ == "__main__":
    path1 = r'F:\data\PTB-TIR\crop511\tirsequences\crowd3\000000.00.z.jpg'
    path2 = r'F:\data\PTB-TIR\crop511\tirsequences\crowd3\000001.00.z.jpg'
    path3 = r'F:\data\PTB-TIR\crop511\tirsequences\crowd3\000002.00.z.jpg'

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img3 = cv2.imread(path3)
    img33 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img22 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img11 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img4 = np.append(img11, img22)
    img4 = img11 + img22 + img33
    cv2.imshow('img1_gray', img4)
    print(img1)
    net = Graph_Attention_Union(3, 3)

    img1 = [img1]
    img2 = [img2]
    img3 = [img3]
    # print(img3.shape)
    img1 = torch.Tensor(img1).permute(0, 3, 1, 2)
    img2 = torch.Tensor(img2).permute(0, 3, 1, 2)
    img3 = torch.Tensor(img3).permute(0, 3, 1, 2)
    x1 = net(img2, img1)
    # x1 = net(x1, img3)
    print(x1.shape)

    x1 = x1[0]
    print(x1.shape)
    x1 = x1.permute(1,2,0)
    x1 = x1.detach().numpy()
    cv2.imshow('11',x1)
    print(x1.shape)
    print(x1)
    dst = cv2.cvtColor(x1, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', dst)
    cv2.waitKey(10000)
    # cv2.imshow('gat', x)