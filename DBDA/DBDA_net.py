import torch
from torch import nn
import math
import numpy as np


import sys
sys.path.append('../global_module/')


class DBDA_network_with_2D(nn.Module):
    def __init__(self, band, classes):
        super(DBDA_network_with_2D, self).__init__()

        # spectral branch

        self.conv11 = nn.Conv2d(in_channels=band, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        # Dense block
        self.batch_norm11 = nn.Sequential(
                                    nn.BatchNorm2d(24), # 动量默认值为0.1
                                    nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Conv2d(in_channels=24, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        self.batch_norm12 = nn.Sequential(
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Conv2d(in_channels=48, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        self.batch_norm13 = nn.Sequential(
                                    nn.BatchNorm2d(72),
                                    nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Conv2d(in_channels=72, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        self.batch_norm14 = nn.Sequential(
                                    nn.BatchNorm2d(96),
                                    nn.ReLU(inplace=True)
        )
        self.conv15 = nn.Conv2d(in_channels=96, out_channels=60, padding=0,
                                kernel_size=1, stride=1) # kernel size随数据变化

        #注意力机制模块

        #self.max_pooling1 = nn.MaxPool3d(kernel_size=(7, 7, 1))
        #self.avg_pooling1 = nn.AvgPool3d(kernel_size=(7, 7, 1))
        self.max_pooling1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pooling1 = nn.AdaptiveAvgPool2d(1)

        self.shared_mlp = nn.Sequential(
                                    nn.Conv3d(in_channels=60, out_channels=30,
                                            kernel_size=(1, 1, 1), stride=(1, 1, 1)),
                                    nn.Conv3d(in_channels=30, out_channels=60,
                                            kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        )

        self.activation1 = nn.Sigmoid()


        # Spatial Branch
        self.conv21 = nn.Conv2d(in_channels=band, out_channels=24,
                                kernel_size=1, stride=1)
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm2d(24),
                                    nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv2d(in_channels=24, out_channels=12, padding=1,
                                kernel_size=3, stride=1)
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm2d(36),
                                    nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv2d(in_channels=36, out_channels=12, padding=1,
                                kernel_size=3, stride=1)
        self.batch_norm23 = nn.Sequential(
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(inplace=True)
        )
        self.conv24 = nn.Conv2d(in_channels=48, out_channels=12, padding=1,
                                kernel_size=3, stride=1)

        # 注意力机制模块

        # self.max_pooling2 = nn.MaxPool3d(kernel_size=(1, 1, 60))
        # self.avg_pooling2 = nn.AvgPool3d(kernel_size=(1, 1, 60))
        # self.max_pooling2 = nn.AdaptiveAvgPool3d(1)
        # self.avg_pooling2 = nn.AdaptiveAvgPool3d(1)

        self.conv25 = nn.Sequential(
                                nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                                kernel_size=(3, 3, 2), stride=(1, 1, 1)),
                                nn.Sigmoid()
        )

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.full_connection = nn.Sequential(
                                # nn.Dropout(p=0.5),
                                nn.Linear(120, classes) # ,
                                # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)

        #fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):
        # spectral
        x11 = self.conv11(X)
        #print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        #print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        #print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        #print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        #print('x16', x16.shape)  # 7*7*97, 60

        # print('x16', x16.shape)
        # 光谱注意力通道
        x1 = self.attention_spectral(x16)
        # print('x1', x1.shape)
        # x1 = torch.mul(x1, x16)


        # spatial
        #print('x', X.shape)
        #X: batch, band, H, W
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        #print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        # print('x25', x25.shape)

        # 空间注意力机制
        x2 = self.attention_spatial(x25)
        # print('x2',x2.shape)
        # x2 = torch.mul(x2, x25)

        # model1
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2= self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)
        #print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        #
        #
        # x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_connection(x_pre)
        # output = self.fc(x_pre)
        return output
# 光谱注意力引导空间信息？ 不行
class DBDA_network_without_attention(nn.Module):#One of the Cross Attention
    def __init__(self, band, classes):
        super(DBDA_network_without_attention, self).__init__()

        # spectral branch

        self.conv11 = nn.Conv2d(in_channels=band, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        # Dense block
        self.batch_norm11 = nn.Sequential(
                                    nn.BatchNorm2d(24),
                                    nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Conv2d(in_channels=24, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        self.batch_norm12 = nn.Sequential(
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Conv2d(in_channels=48, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        self.batch_norm13 = nn.Sequential(
                                    nn.BatchNorm2d(72),
                                    nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Conv2d(in_channels=72, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        self.batch_norm14 = nn.Sequential(
                                    nn.BatchNorm2d(96),
                                    nn.ReLU(inplace=True)
        )
        self.conv15 = nn.Conv2d(in_channels=96, out_channels=60, padding=0,
                                kernel_size=1, stride=1) # kernel size随数据变化

        #注意力机制模块

        #self.max_pooling1 = nn.MaxPool3d(kernel_size=(7, 7, 1))
        #self.avg_pooling1 = nn.AvgPool3d(kernel_size=(7, 7, 1))
        self.max_pooling1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pooling1 = nn.AdaptiveAvgPool2d(1)

        self.shared_mlp = nn.Sequential(
                                    nn.Conv3d(in_channels=60, out_channels=30,
                                            kernel_size=(1, 1, 1), stride=(1, 1, 1)),
                                    nn.Conv3d(in_channels=30, out_channels=60,
                                            kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        )

        self.activation1 = nn.Sigmoid()


        # Spatial Branch
        self.conv21 = nn.Conv2d(in_channels=band, out_channels=24,
                                kernel_size=1, stride=1)
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm2d(24),
                                    nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv2d(in_channels=24, out_channels=12, padding=1,
                                kernel_size=3, stride=1)
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm2d(36),
                                    nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv2d(in_channels=36, out_channels=12, padding=1,
                                kernel_size=3, stride=1)
        self.batch_norm23 = nn.Sequential(
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(inplace=True)
        )
        self.conv24 = nn.Conv2d(in_channels=48, out_channels=12, padding=1,
                                kernel_size=3, stride=1)

        # 注意力机制模块

        # self.max_pooling2 = nn.MaxPool3d(kernel_size=(1, 1, 60))
        # self.avg_pooling2 = nn.AvgPool3d(kernel_size=(1, 1, 60))
        # self.max_pooling2 = nn.AdaptiveAvgPool3d(1)
        # self.avg_pooling2 = nn.AdaptiveAvgPool3d(1)

        self.conv25 = nn.Sequential(
                                nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                                kernel_size=(3, 3, 2), stride=(1, 1, 1)),
                                nn.Sigmoid()
        )

        # self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.global_pooling = nn.AvgPool2d(kernel_size=(11,11),padding=0,stride=1)#5
        self.full_connection = nn.Sequential(nn.Conv2d(in_channels=120, out_channels=classes, padding=0,kernel_size=1, stride=1)
                                # nn.Dropout(p=0.5),
                                #nn.Linear(120, classes) # ,
                                # nn.Softmax()
        )

        self.attention_spectral = CAM_Module_Cross()#CAM_Module(60)

        self.attention_spatial = PAM_Module(60)

        #fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):
        # spectral
        x11 = self.conv11(X)
        #print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        #print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        #print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        #print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        #print('x16', x16.shape)  # 7*7*97, 60

        # print('x16', x16.shape)
        # 光谱注意力通道
        # x1 = self.attention_spectral(x16)
        x1 = x16
        # print('x1', x1.shape)
        # x1 = torch.mul(x1, x16)

        X = self.attention_spectral(X,X)
        # spatial
        #print('x', X.shape)
        #X: batch, band, H, W
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        #print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        # print('x25', x25.shape)

        # 空间注意力机制
        # x2 = self.attention_spatial(x25)
        x2 = x25
        # print('x2',x2.shape)
        # x2 = torch.mul(x2, x25)


        # model1
        x1 = self.global_pooling(x1)
        # x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2= self.global_pooling(x2)
        # x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)
        #print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        #
        #
        # x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_connection(x_pre)
        output = output.squeeze(-1).squeeze(-1).squeeze(-1)
        return output

class DBDA_network_three_losses(nn.Module):#No Attention, three losses
    def __init__(self, band, classes):
        super(DBDA_network_three_losses, self).__init__()

        # spectral branch

        self.conv11 = nn.Conv2d(in_channels=band, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        # Dense block
        self.batch_norm11 = nn.Sequential(
                                    nn.BatchNorm2d(24), # 动量默认值为0.1
                                    nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Conv2d(in_channels=24, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        self.batch_norm12 = nn.Sequential(
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Conv2d(in_channels=48, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        self.batch_norm13 = nn.Sequential(
                                    nn.BatchNorm2d(72),
                                    nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Conv2d(in_channels=72, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        self.batch_norm14 = nn.Sequential(
                                    nn.BatchNorm2d(96),
                                    nn.ReLU(inplace=True)
        )
        self.conv15 = nn.Conv2d(in_channels=96, out_channels=60, padding=0,
                                kernel_size=1, stride=1) # kernel size随数据变化

        #注意力机制模块

        #self.max_pooling1 = nn.MaxPool3d(kernel_size=(7, 7, 1))
        #self.avg_pooling1 = nn.AvgPool3d(kernel_size=(7, 7, 1))
        self.max_pooling1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pooling1 = nn.AdaptiveAvgPool2d(1)

        self.shared_mlp = nn.Sequential(
                                    nn.Conv3d(in_channels=60, out_channels=30,
                                            kernel_size=(1, 1, 1), stride=(1, 1, 1)),
                                    nn.Conv3d(in_channels=30, out_channels=60,
                                            kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        )

        self.activation1 = nn.Sigmoid()


        # Spatial Branch
        self.conv21 = nn.Conv2d(in_channels=band, out_channels=24,
                                kernel_size=1, stride=1)
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm2d(24),
                                    nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv2d(in_channels=24, out_channels=12, padding=1,
                                kernel_size=3, stride=1)
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm2d(36),
                                    nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv2d(in_channels=36, out_channels=12, padding=1,
                                kernel_size=3, stride=1)
        self.batch_norm23 = nn.Sequential(
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(inplace=True)
        )
        self.conv24 = nn.Conv2d(in_channels=48, out_channels=12, padding=1,
                                kernel_size=3, stride=1)

        # 注意力机制模块

        # self.max_pooling2 = nn.MaxPool3d(kernel_size=(1, 1, 60))
        # self.avg_pooling2 = nn.AvgPool3d(kernel_size=(1, 1, 60))
        # self.max_pooling2 = nn.AdaptiveAvgPool3d(1)
        # self.avg_pooling2 = nn.AdaptiveAvgPool3d(1)

        self.conv25 = nn.Sequential(
                                nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                                kernel_size=(3, 3, 2), stride=(1, 1, 1)),
                                nn.Sigmoid()
        )

        # self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.global_pooling = nn.AvgPool2d(kernel_size=(11,11),padding=0,stride=1)#5
        self.full_connection = nn.Sequential(nn.Conv2d(in_channels=120, out_channels=60, padding=0,kernel_size=1, stride=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(in_channels=60, out_channels=classes, padding=0, kernel_size=1,
                                                       stride=1)
                                # nn.Dropout(p=0.5),
                                #nn.Linear(120, classes) # ,
                                # nn.Softmax()
        )
        self.full_connection_spec = nn.Sequential(nn.Conv2d(in_channels=60, out_channels=classes, padding=0, kernel_size=1, stride=1))
        self.full_connection_spat = nn.Sequential(
            nn.Conv2d(in_channels=60, out_channels=classes, padding=0, kernel_size=1, stride=1))


        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)

        #fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):
        # spectral
        x11 = self.conv11(X)
        #print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        #print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        #print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        #print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        #print('x16', x16.shape)  # 7*7*97, 60

        # print('x16', x16.shape)
        # 光谱注意力通道
        # x1 = self.attention_spectral(x16)
        x1 = x16
        # print('x1', x1.shape)
        # x1 = torch.mul(x1, x16)


        # spatial
        #print('x', X.shape)
        #X: batch, band, H, W
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        #print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        # print('x25', x25.shape)

        # 空间注意力机制
        # x2 = self.attention_spatial(x25)
        x2 = x25
        # print('x2',x2.shape)
        # x2 = torch.mul(x2, x25)


        # model1
        x1 = self.global_pooling(x1)
        output_spec = self.full_connection_spec(x1)
        # x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2= self.global_pooling(x2)
        # x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)
        output_spat = self.full_connection_spat(x2)
        x_pre = torch.cat((x1, x2), dim=1)
        #print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        #
        #
        # x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_connection(x_pre)
        output = output.squeeze(-1).squeeze(-1).squeeze(-1)
        output_spec = output_spec.squeeze(-1).squeeze(-1).squeeze(-1)
        output_spat = output_spat.squeeze(-1).squeeze(-1).squeeze(-1)
        return output, output_spec, output_spat


class DBDA_network_three_losses_cross_Attention(nn.Module):#No Attention, three losses
    def __init__(self, band, classes):
        super(DBDA_network_three_losses_cross_Attention, self).__init__()

        # spectral branch

        self.conv11 = nn.Conv2d(in_channels=band, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        # Dense block
        self.batch_norm11 = nn.Sequential(
                                    nn.BatchNorm2d(24), # 动量默认值为0.1
                                    nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Conv2d(in_channels=24, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        self.batch_norm12 = nn.Sequential(
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Conv2d(in_channels=48, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        self.batch_norm13 = nn.Sequential(
                                    nn.BatchNorm2d(72),
                                    nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Conv2d(in_channels=72, out_channels=24, padding=0,
                                kernel_size=1, stride=1)
        self.batch_norm14 = nn.Sequential(
                                    nn.BatchNorm2d(96),
                                    nn.ReLU(inplace=True)
        )
        self.conv15 = nn.Conv2d(in_channels=96, out_channels=60, padding=0,
                                kernel_size=1, stride=1) # kernel size随数据变化

        #注意力机制模块

        #self.max_pooling1 = nn.MaxPool3d(kernel_size=(7, 7, 1))
        #self.avg_pooling1 = nn.AvgPool3d(kernel_size=(7, 7, 1))
        self.max_pooling1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pooling1 = nn.AdaptiveAvgPool2d(1)

        self.shared_mlp = nn.Sequential(
                                    nn.Conv3d(in_channels=60, out_channels=30,
                                            kernel_size=(1, 1, 1), stride=(1, 1, 1)),
                                    nn.Conv3d(in_channels=30, out_channels=60,
                                            kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        )

        self.activation1 = nn.Sigmoid()


        # Spatial Branch
        self.conv21 = nn.Conv2d(in_channels=band, out_channels=24,
                                kernel_size=1, stride=1)
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm2d(24),
                                    nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv2d(in_channels=24, out_channels=12, padding=1,
                                kernel_size=3, stride=1)
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm2d(36),
                                    nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv2d(in_channels=36, out_channels=12, padding=1,
                                kernel_size=3, stride=1)
        self.batch_norm23 = nn.Sequential(
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(inplace=True)
        )
        self.conv24 = nn.Conv2d(in_channels=48, out_channels=12, padding=1,
                                kernel_size=3, stride=1)

        # 注意力机制模块

        # self.max_pooling2 = nn.MaxPool3d(kernel_size=(1, 1, 60))
        # self.avg_pooling2 = nn.AvgPool3d(kernel_size=(1, 1, 60))
        # self.max_pooling2 = nn.AdaptiveAvgPool3d(1)
        # self.avg_pooling2 = nn.AdaptiveAvgPool3d(1)

        self.conv25 = nn.Sequential(
                                nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                                kernel_size=(3, 3, 2), stride=(1, 1, 1)),
                                nn.Sigmoid()
        )

        # self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.global_pooling = nn.AvgPool2d(kernel_size=(11,11),padding=0,stride=1)#5
        self.full_connection = nn.Sequential(nn.Conv2d(in_channels=120, out_channels=60, padding=0,kernel_size=1, stride=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(in_channels=60, out_channels=classes, padding=0, kernel_size=1,
                                                       stride=1)
                                # nn.Dropout(p=0.5),
                                #nn.Linear(120, classes) # ,
                                # nn.Softmax()
        )
        self.full_connection_spec = nn.Sequential(nn.Conv2d(in_channels=60, out_channels=classes, padding=0, kernel_size=1, stride=1))
        self.full_connection_spat = nn.Sequential(
            nn.Conv2d(in_channels=60, out_channels=classes, padding=0, kernel_size=1, stride=1))


        self.attention_spectral = CAM_Module_Cross()
        self.attention_spatial = PAM_Module(60)

        #fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):
        # spectral
        x11 = self.conv11(X)
        #print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        #print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        #print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        #print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        #print('x16', x16.shape)  # 7*7*97, 60

        # print('x16', x16.shape)
        # 光谱注意力通道
        # x1 = self.attention_spectral(x16)
        x1 = x16
        # print('x1', x1.shape)
        # x1 = torch.mul(x1, x16)
        X = self.attention_spectral(X, X)

        # spatial
        #print('x', X.shape)
        #X: batch, band, H, W
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        #print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        # print('x25', x25.shape)

        # 空间注意力机制
        # x2 = self.attention_spatial(x25)
        x2 = x25
        # print('x2',x2.shape)
        # x2 = torch.mul(x2, x25)


        # model1
        x1 = self.global_pooling(x1)
        output_spec = self.full_connection_spec(x1)
        # x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2= self.global_pooling(x2)
        # x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)
        output_spat = self.full_connection_spat(x2)
        x_pre = torch.cat((x1, x2), dim=1)
        #print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        #
        #
        # x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_connection(x_pre)
        output = output.squeeze(-1).squeeze(-1).squeeze(-1)
        output_spec = output_spec.squeeze(-1).squeeze(-1).squeeze(-1)
        output_spat = output_spat.squeeze(-1).squeeze(-1).squeeze(-1)
        return output, output_spec, output_spat

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # self.query_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.value_conv = Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, -1, width * height)
        # proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = (self.gamma*out + x)#.unsqueeze(-1)
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        #print(x.size())
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) #形状转换并交换维度
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        # print('out', out.shape)
        # print('x', x.shape)

        out = self.gamma*out + x  #C*H*W
        return out

class CAM_Module_Cross(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(CAM_Module_Cross, self).__init__()
        self.Feature_Extractor = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=5, padding=(1, 1, 0),
                  kernel_size=(3, 3, 1), stride=(1, 1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(in_channels=5, out_channels=10, padding=(1, 1, 0),
                                              kernel_size=(3, 3, 1), stride=(1, 1, 1))
                                    )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x,proj_value):
        m_batchsize, C, height, width = x.size()
        x = x.permute(0,2,3,1).unsqueeze(1)
        # tmp = self.Feature_Extractor(x).squeeze()
        tmp = self.Feature_Extractor(x)
        print(tmp.shape)  # batch,channal,W,H,band
        # tmp = tmp.permute(0,2,3,4,1)
        proj_query = tmp.reshape(m_batchsize * height * width, C, -1)
        # print(proj_query.shape)
        proj_key = tmp.reshape(m_batchsize * height * width, C, -1).permute(0, 2, 1) #形状转换并交换维度
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy#
        attention = self.softmax(energy_new)
        proj_value = proj_value.permute(0,2,3,1).reshape(m_batchsize * height * width, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, height, width, C).permute(0,3,1,2)
        # print('out', out.shape)
        # print('x', x.shape)

        # out = self.gamma*out + x  #C*H*W
        return out

if __name__ == '__main__':
    # model1 = DBDA_network_without_attention(11, 2).cuda()
    # model2 = DBDA_network_three_losses(11, 2).cuda()
    model3 = DBDA_network_three_losses_cross_Attention(11, 2).cuda()
    input = torch.rand(1, 11, 1415, 1859).cuda()
    with torch.no_grad():
        # predict1 = model1(input)
        # predict2 = model2(input)
        predict3 = model3(input)
        # print(predict1.shape)
        # print(predict2[0].shape)
        print(predict3[0].shape)