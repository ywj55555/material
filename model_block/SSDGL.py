import sys
sys.path.append('../')
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import scipy.io as io
import configparser
import warnings
warnings.filterwarnings("ignore")
from model_block.criterion import OhemCrossEntropy


def conv3x3_gn_relu(in_channel, out_channel, num_group):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.GroupNorm(num_group, out_channel),
        nn.ReLU(inplace=True),
    )


def gn_relu(in_channel, num_group):
    return nn.Sequential(
        nn.GroupNorm(num_group, in_channel),
        nn.ReLU(inplace=True),
    )


def downsample2x(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 2, 1),
        nn.ReLU(inplace=True)
    )

# 96 8 1 3
def repeat_block(block_channel1, r, n, conv_size):
    cl_channel = block_channel1 / 8  # 12
    cl_channel = int(cl_channel)  # 12
    cl2_channel = int(cl_channel / 2)  # 6
    gn_a = int(block_channel1 / 2)  # 48
    layers = [
        nn.Sequential(  # 12                                     12           6                         3
            ConvLSTM(input_channels=cl_channel, hidden_channels=[cl_channel, cl2_channel], kernel_size=conv_size,
                     step=8,
                     effective_step=[7]).cuda(),
            BasicBlock(gn_a), gn_relu(block_channel1, r), )]
    return nn.Sequential(*layers)

class SSDGL(nn.Module):
    def __init__(self, bands=9, class_nums=4):
        super(SSDGL, self).__init__()
        # if cfg.data.train.params.select_type == 'sample_percent':
        self.reduction_ratio = 1.0
        self.block_channels = [96, 128, 192, 256]
        r = int(8 * 1.0)
        kernel_size = 3
        self.in_channels = bands
        self.num_classes = class_nums
        self.inner_dim = 128
        self.num_blocks = [1, 1, 1, 1]
        # else:
        #     r = int(4 * self.config.reduction_ratio)
        #     kernel_size = 5

        # block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        # block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        # block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        # block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        block1_channels = int(self.block_channels[0] * self.reduction_ratio / r) * r  # 96
        block2_channels = int(self.block_channels[1] * self.reduction_ratio / r) * r  # 128
        block3_channels = int(self.block_channels[2] * self.reduction_ratio / r) * r
        block4_channels = int(self.block_channels[3] * self.reduction_ratio / r) * r

        self.feature_ops = nn.ModuleList([

            conv3x3_gn_relu(self.in_channels, block1_channels, r),
            # 96 8 1 3
            repeat_block(block1_channels, r, self.num_blocks[0], kernel_size),  # num_blocks=(1, 1, 1, 1)
            nn.Identity(),

            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.num_blocks[1], kernel_size),
            nn.Identity(),

            downsample2x(block2_channels, block3_channels),
            repeat_block(block3_channels, r, self.num_blocks[2], kernel_size),
            nn.Identity(),

            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.num_blocks[3], kernel_size),
            nn.Identity(),
        ])
        inner_dim = int(self.inner_dim * self.reduction_ratio)

        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        # if cfg.data.train.params.select_type == 'sample_percent':
        self.SA = nn.ModuleList([
            SpatialAttention(),
            SpatialAttention(),
            SpatialAttention(),
            SpatialAttention(),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            nn.Conv2d(inner_dim, self.in_channels, 3, 1, 1),
        ])

        self.cls_pred_conv = nn.Conv2d(self.in_channels, self.num_classes, 1)

    def top_down(self, top, lateral):

        top2x = F.interpolate(top, scale_factor=2.0, mode='bilinear')

        return lateral + top2x

    def forward(self, x):
        feat_list = []

        for op in self.feature_ops:
            x = op(x)

            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]

        inner_feat_list.reverse()
        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(inner_feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i + 1])

            out = self.fuse_3x3convs[i + 1](inner)

            out_feat_list.append(out)
        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        return logit
        # if self.training:
        #     loss_dict = {
        #         'cls_loss': self.loss(logit, y.cuda(), w)
        #     }
        #     return loss_dict
        #
        # return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        print("weight", weight.sum())

        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.sum() / weight.sum()
        # v = losses.mul_(weight).sum() / weight.sum()
        return v

    # def set_defalut_config(self):
    #     # pavia
    #     self.config.update(dict(
    #         in_channels=103,
    #         num_classes=9,
    #         block_channels=(96, 128, 192, 256),
    #         num_blocks=(1, 1, 1, 1),
    #         inner_dim=128,
    #         reduction_ratio=1.0,
    #     ))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # ratio = 24 if dataset_path == "SSDGL.SSDGL_1_0_Indianpine" else 16
        ratio = 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)

        self.relu1 = nn.GELU()
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        residual = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))

        # avg_out = self.relu2(avg_out)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        # max_out = self.relu2(max_out)
        out = self.relu2(avg_out + max_out)

        y = x * out.view(out.size(0), out.size(1), 1, 1)

        y = y + residual
        return y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        # self.relu1 = nn.GELU()
        self.relu1 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)
        out1 = self.conv1(out)

        out2 = self.relu1(out1)

        out = self.sigmoid(out2)

        y = x * out.view(out.size(0), 1, out.size(-2), out.size(-1))
        y = y + residual
        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, planes):
        super(BasicBlock, self).__init__()

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # out = self.ca(x) + self.sa(x)
        out = torch.cat([self.ca(x), self.sa(x)], dim=1)
        # min -

        return out


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None
        # self.relu = nn.ReLU(inplace=True)
        # self.relu1 = nn.GELU()

    def forward(self, x, h, c):

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)

        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):

        # print(input.size())
        # 20 96, 352, 624
        internal_state = []
        outputs = []

        for step in range(self.step):
            # a = input.squeeze()

            # b = int(len(a) / 8)  # 和批量有关！！
            b = int(input.size(1) / 8)  # 和批量有关！！

            x = input[:, step * b:(step + 1) * b, :, :]  # 通道分为8组

            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            deng = self.effective_step[0]
            if step <= deng:
                outputs.append(x)
        result = outputs[0]

        for i in range(self.step - 1):
            result = torch.cat([result, outputs[i + 1]], dim=1)
        return result

from torch.cuda.amp import autocast, GradScaler

if __name__ == '__main__':
    model = SSDGL(9, 4).cuda()
    labelNp = np.random.randint(0, 4, (2, 1008, 1008))
    label = torch.tensor(labelNp).cuda().long()
    spectralData = torch.rand(2, 9, 1008, 1008).cuda()  # 为什么和batch有关？？
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = OhemCrossEntropy(ignore_label=-1,
                                 thres=0.9,
                                 min_kept=500000,
                                 weight=None,
                                 model_num_outputs=1,
                                 loss_balance_weights=[1])
    # criterion =
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = GradScaler()
    model.train()
    for epoch in range(1000):
        with autocast():
            predict = model(spectralData)
            # print(predict.size())
        # losses = F.cross_entropy(predict, label, weight=None,
        #                          ignore_index=-1, reduction='none')
        losses = criterion(predict, label)
        loss = losses.mean()
        # loss = criterion(predict, label)  # + L2Loss(model, 1e-4)
        print(epoch)
        print(loss)
        # trainLossTotal += loss
        # print("loss = %.5f" % float(loss))
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        model.zero_grad()
        scaler.scale(loss).backward()
        # trainLossTotal += loss.item()
        scaler.step(optimizer)
        scaler.update()

        # predict = predict.squeeze()
        predictIndex = torch.argmax(predict, dim=1)  # 计算一下准确率和召回率 B*H*W 和label1一样
        count_right = torch.sum(predictIndex == label).item()  # 全部训练
        count_tot = torch.sum(label != -1).item()
        print(count_right / count_tot)

    print("successfully")
    # print(output.size())





