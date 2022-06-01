import numpy as np
import torch
from torch import nn
import cv2
from torch.nn import init
from pysot.models.attention.GAT import Graph_Attention_Union

def spatial_shift1(x):
    # b, w, h, c = x.size()
    b, w, h, c = x.shape
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    # b, w, h, c = x.size()
    b, w, h, c = x.shape
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self, channel, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)  # bs,k,n,c
        a = torch.sum(torch.sum(x_all, 1), 1)  # bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # bs,kc
        hat_a = hat_a.reshape(b, self.k, c)  # bs,k,c
        bar_a = self.softmax(hat_a)  # bs,k,c
        attention = bar_a.unsqueeze(-2)  # #bs,k,1,c
        out = attention * x_all  # #bs,k,n,c
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


class S2Attention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)

        self.split_attention = SplitAttention(channels)
        self.GAT = Graph_Attention_Union(256, 256)

    def forward(self, x, z):
        def shift(x):
            b, c, w, h = x.size()
            x = x.permute(0, 2, 3, 1)
            x = self.mlp1(x)
            x1 = spatial_shift1(x[:, :, :, :c])
            # x1 = x[:, :, :, :c]
            x2 = spatial_shift2(x[:, :, :, c:c * 2])
            # 消融实验 不使用shift2
            # x2 = x[:, :, :, c:c * 2]
            x3 = x[:, :, :, c * 2:]
            return x1, x2, x3


        x1, x2, x3 = shift(x)
        z1, z2, z3 = shift(z)

        x = x.permute(0, 2, 3, 1)
        z = z.permute(0, 2, 3, 1)
        x1, x2, x3 = x1 + x, x2 + x, x3 + x
        z1, z2, z3 = z1 + z, z2 + z, z3 + z

        x1 = self.GAT(z1.permute(0, 3, 1, 2), x1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x2 = self.GAT(z2.permute(0, 3, 1, 2), x2.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x3 = self.GAT(z3.permute(0, 3, 1, 2), x3.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        # 消融实验 看没有split attention效果怎么样，不好
        # a = x1 + x2 + x3

        # 消融实验 只要x1 + x2
        # x_all = torch.stack([x1, x2], 1)
        # 消融实验 只要x1 + x3
        # x_all = torch.stack([x1, x3], 1)
        # 消融实验 只要x2 + x3
        # x_all = torch.stack([x2, x3], 1)
        # a = self.split_attention(x_all)

        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)
        return x


if __name__ == '__main__':
    input1 = torch.randn(8, 256, 25, 25)
    input2 = torch.randn(8, 256, 13, 13)
    img = cv2.imread(r"G:\my_data\LSOTB-TIR\Evaluation Dataset\sequences\bus_V_001\img\0099.jpg")
    img = np.array([img])
    print(type(img))
    img1 = spatial_shift1(img)
    img2 = spatial_shift2(img)
    # print(img1.shape)
    img1 = img1[0]
    img2 = img2[0]
    img = img[0]
    cv2.imshow("img", img1)
    # cv2.imshow("img",img)
    cv2.waitKey(10000)
    s2att = S2Attention(channels=256)
    output = s2att(input1, input2)
    print(output.shape)