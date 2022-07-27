# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.append('../')
from model_block.UAFM import UAFM_ChAtten


class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 bias = False,
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2 if padding else 0,
            bias = bias, **kwargs)
        self._batch_norm = nn.BatchNorm2d(out_channels, momentum=0.1)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride = 1,
                 padding=1,
                 bias = False,
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2 if padding else 0, bias = bias,**kwargs)

        self._batch_norm = nn.BatchNorm2d(out_channels, momentum=0.1)
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


class ConvBNRelu(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride = 1,
                 padding=1,
                 bias = False,
                 **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2 if padding else 0, bias = bias, **kwargs)

        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def avg_max_reduce_channel_helper(x, use_concat=True):
    # Reduce hw by avg and max, only support single input
    assert not isinstance(x, (list, tuple))
    # print("x before mean and max:", x.shape)
    mean_value = torch.mean(x, dim=1, keepdim=True)
    max_value = torch.max(x, dim=1, keepdim=True)[0]
    # mean_value = mean_value.unsqueeze(0)
    # print("mean max:", mean_value.shape, max_value.shape)

    if use_concat:
        res = torch.at([mean_value, max_value], dim=1)
    else:
        res = [mean_value, max_value]
    return res


def avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            # print(xi.shape)
            res.extend(avg_max_reduce_channel_helper(xi, False))
        # print("res:\n",)
        # for it in res:
        #     print(it.shape)
        return torch.cat(res, dim=1)


class UAFM(nn.Module):
    """
    The base of Unified Attention Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='nearest'):
        super().__init__()

        self.conv_x = ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias=False)
        self.conv_out = ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.resize_mode = resize_mode

    def check(self, x, y):
        # print("x dim:",x.ndim)
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        y_up = F.interpolate(y, x.shape[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        # print("x,y shape:",x.shape, y.shape)
        self.check(x, y)
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out


class UAFM_SpAtten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """
    # 没有使用 通道 attention
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='nearest'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvBNReLU(
                4, 2, kernel_size=3, padding=1, bias=False),
            ConvBN(
                2, 1, kernel_size=3, padding=1, bias=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        # print("x, y shape:",x.shape, y.shape)
        atten = avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out




class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias=False),
                nn.BatchNorm2d(out_planes // 2, momentum=0.1), )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(
                        in_planes, out_planes // 2, kernel_size=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes
                               // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes
                               // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = torch.cat(out_list, dim=1)
        return out


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias=False),
                nn.BatchNorm2D(out_planes // 2, momentum=0.1), )
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    in_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_planes,
                    bias_attr=False),
                nn.BatchNorm2d(in_planes, momentum=0.1),
                nn.Conv2d(
                    in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes, momentum=0.1), )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(
                        in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes
                               // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes
                               // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x
        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)
        if self.stride == 2:
            x = self.skip(x)
        return torch.cat(out_list, dim=1) + x


class STDCNet(nn.Module):
    """
    The STDCNet implementation based on Pytorch.

    The original article refers to Meituan
    Fan, Mingyuan, et al. "Rethinking BiSeNet For Real-time Semantic Segmentation."
    (https://arxiv.org/abs/2104.13188)

    Args:
        base(int, optional): base channels. Default: 64.
        layers(list, optional): layers numbers list. It determines STDC block numbers of STDCNet's stage3\4\5. Defualt: [4, 5, 3].
        block_num(int,optional): block_num of features block. Default: 4.
        type(str,optional): feature fusion method "cat"/"add". Default: "cat".
        num_classes(int, optional): class number for image classification. Default: 1000.
        dropout(float,optional): dropout ratio. if >0,use dropout ratio.  Default: 0.20.
        use_conv_last(bool,optional): whether to use the last ConvBNReLU layer . Default: False.
        pretrained(str, optional): the path of pretrained model.
    """

    def __init__(self,
                 base=64,
                 layers=[4, 5, 3],
                 block_num=4,
                 type="cat",
                 num_classes=1000,
                 dropout=0.20,
                 use_conv_last=False,
                 pretrained=None,
                 input_channel=3):
        super(STDCNet, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.input_channel = input_channel
        self.use_conv_last = use_conv_last
        self.feat_channels = [base // 2, base, base * 4, base * 8, base * 16]
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvBNRelu(base * 16, max(1024, base * 16), 1, 1)


        if (layers == [4, 5, 3]):  # stdc1446
            self.x2 = nn.Sequential(self.features[:1])
            self.x4 = nn.Sequential(self.features[1:2])
            self.x8 = nn.Sequential(self.features[2:6])
            self.x16 = nn.Sequential(self.features[6:11])
            self.x32 = nn.Sequential(self.features[11:])
        elif (layers == [2, 2, 2]):  # stdc813
            self.x2 = nn.Sequential(self.features[:1])
            self.x4 = nn.Sequential(self.features[1:2])
            self.x8 = nn.Sequential(self.features[2:4])
            self.x16 = nn.Sequential(self.features[4:6])
            self.x32 = nn.Sequential(self.features[6:])
        else:
            raise NotImplementedError(
                "model with layers:{} is not implemented!".format(layers))

        self.pretrained = pretrained
        # self.init_weight()

    def forward(self, x):
        """
        forward function for feature extract.
        """
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)
        return feat2, feat4, feat8, feat16, feat32

    def _make_layers(self, base, layers, block_num, block):
        features = []
        # stride = 2 达到降采样目的
        features += [ConvBNRelu(self.input_channel, base // 2, 3, 2)]
        features += [ConvBNRelu(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    # 最后一个参数为 stride = 2 ，尺寸减半，就是每个layer 的第一层尺寸先减半
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(
                        block(base * int(math.pow(2, i + 1)), base * int(
                            math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(
                        block(base * int(math.pow(2, i + 2)), base * int(
                            math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    # def init_weight(self):
    #     for layer in self.sublayers():
    #         if isinstance(layer, nn.Conv2D):
    #             param_init.normal_init(layer.weight, std=0.001)
    #         elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
    #             param_init.constant_init(layer.weight, value=1.0)
    #             param_init.constant_init(layer.bias, value=0.0)
    #     if self.pretrained is not None:
    #         utils.load_pretrained_model(self, self.pretrained)


def STDC2(input_channel=3,**kwargs):
    model = STDCNet(base=64, layers=[4, 5, 3],input_channel=input_channel,**kwargs)
    return model


class PPLiteSeg(nn.Module):
    """
    The PP_LiteSeg implementation based on Pytorch.

    The original article refers to "Juncai Peng, Yi Liu, Shiyu Tang, Yuying Hao, Lutao Chu,
    Guowei Chen, Zewu Wu, Zeyu Chen, Zhiliang Yu, Yuning Du, Qingqing Dang,Baohua Lai,
    Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma. PP-LiteSeg: A Superior Real-Time Semantic
    Segmentation Model. https://arxiv.org/abs/2204.02681".

    Args:
        num_classes (int): The number of target classes.
        backbone(nn.Layer): Backbone network, such as stdc1net and resnet18. The backbone must
            has feat_channels, of which the length is 5.
        backbone_indices (List(int), optional): The values indicate the indices of output of backbone.
            Default: [2, 3, 4].
        arm_type (str, optional): The type of attention refinement module. Default: ARM_Add_SpAttenAdd3.
        cm_bin_sizes (List(int), optional): The bin size of context module. Default: [1,2,4].
        cm_out_ch (int, optional): The output channel of the last context module. Default: 128.
        arm_out_chs (List(int), optional): The out channels of each arm module. Default: [64, 96, 128].
        seg_head_inter_chs (List(int), optional): The intermediate channels of segmentation head.
            Default: [64, 64, 64].
        resize_mode (str, optional): The resize mode for the upsampling operation in decoder.
            Default: bilinear.
        pretrained (str, optional): The path or url of pretrained model. Default: None.

    """

    def __init__(self,
                 num_classes = 19,
                 # backbone = STDC2(),
                 input_channel=3,
                 backbone_indices=[2, 3, 4],
                 arm_type='UAFM_SpAtten',
                 cm_bin_sizes=[1, 2, 4],
                 cm_out_ch=128,
                 arm_out_chs=[64, 96, 128],
                 seg_head_inter_chs=[64, 64, 64],
                 resize_mode='nearest',
                 pretrained=False):
        super().__init__()

        backbone = STDC2(input_channel)
        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should has feat_channels."
        assert len(backbone.feat_channels) >= len(backbone_indices), \
            f"The length of input backbone_indices ({len(backbone_indices)}) should not be" \
            f"greater than the length of feat_channels ({len(backbone.feat_channels)})."
        assert len(backbone.feat_channels) > max(backbone_indices), \
            f"The max value ({max(backbone_indices)}) of backbone_indices should be " \
            f"less than the length of feat_channels ({len(backbone.feat_channels)})."
        self.backbone = backbone

        assert len(backbone_indices) > 1, "The lenght of backbone_indices " \
                                          "should be greater than 1"
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]

        # head
        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
        assert len(arm_out_chs) == len(backbone_indices), "The length of " \
                                                          "arm_out_chs and backbone_indices should be equal"

        self.ppseg_head = PPLiteSegHead(backbone_out_chs, arm_out_chs,
                                        cm_bin_sizes, cm_out_ch, arm_type,
                                        resize_mode)

        if len(seg_head_inter_chs) == 1:
            seg_head_inter_chs = seg_head_inter_chs * len(backbone_indices)
        assert len(seg_head_inter_chs) == len(backbone_indices), "The length of " \
                                                                 "seg_head_inter_chs and backbone_indices should be equal"
        self.seg_heads = nn.ModuleList()  # [..., head_16, head32]
        print("arm_out_chs:",arm_out_chs, " ; seg_head_inter_chs:",seg_head_inter_chs)
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, num_classes))  # to adjust feature

        # pretrained
        self.pretrained = pretrained
        # self.init_weight()

    def forward(self, x):
        x_hw = x.shape[2:]
        # print("x_hw:",x_hw)

        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        # print(type(feats_backbone))
        assert len(feats_backbone) >= len(self.backbone_indices), \
            f"The nums of backbone feats ({len(feats_backbone)}) should be greater or " \
            f"equal than the nums of backbone_indices ({len(self.backbone_indices)})"

        feats_selected = [feats_backbone[i] for i in self.backbone_indices]

        feats_head = self.ppseg_head(feats_selected)  # [..., x8, x16, x32]
        # 看看feats_head的尺寸就好，然后在这里融合
        # for fea in feats_head:
        #     print(fea.shape)

        if self.training:
            logit_list = []

            for x, seg_head in zip(feats_head, self.seg_heads):
                x = seg_head(x)
                logit_list.append(x)

            logit_list = [
                F.interpolate(
                    x, x_hw, mode='bilinear', align_corners=None)
                for x in logit_list
            ]
        else:
            x = self.seg_heads[0](feats_head[0])
            # print("x:",x.shape)
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=None)
            logit_list = [x]

        return logit_list

    # def init_weight(self):
    #     if self.pretrained is not None:
    #         utils.load_entire_model(self, self.pretrained)


class PPLiteSegHead(nn.Module):
    """
    The head of PPLiteSeg.

    Args:
        backbone_out_chs (List(Tensor)): The channels of output tensors in the backbone.
        arm_out_chs (List(int)): The out channels of each arm module.
        cm_bin_sizes (List(int)): The bin size of context module.
        cm_out_ch (int): The output channel of the last context module.
        arm_type (str): The type of attention refinement module.
        resize_mode (str): The resize mode for the upsampling operation in decoder.
    """

    def __init__(self, backbone_out_chs, arm_out_chs, cm_bin_sizes, cm_out_ch,
                 arm_type, resize_mode):
        super().__init__()

        self.cm = PPContextModule(backbone_out_chs[-1], cm_out_ch, cm_out_ch,
                                  cm_bin_sizes)

        # assert hasattr(layers, arm_type), \
        #     "Not support arm_type ({})".format(arm_type)
        arm_class = eval(arm_type)

        self.arm_list = nn.ModuleList()  # [..., arm8, arm16, arm32]
        for i in range(len(backbone_out_chs)):
            low_chs = backbone_out_chs[i]
            high_ch = cm_out_ch if i == len(
                backbone_out_chs) - 1 else arm_out_chs[i + 1]
            out_ch = arm_out_chs[i]
            arm = arm_class(
                low_chs, high_ch, out_ch, ksize=3, resize_mode=resize_mode)
            self.arm_list.append(arm)

    def forward(self, in_feat_list):
        """
        Args:
            in_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
        Returns:
            out_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
                The length of in_feat_list and out_feat_list are the same.
        """
        # 可选 提取空间上下文信息的 特征层级，默认只提取最高一层级，输出shape 和 输入 shape 一致！！！
        high_feat = self.cm(in_feat_list[-1])
        out_feat_list = []

        for i in reversed(range(len(in_feat_list))):
            low_feat = in_feat_list[i]
            arm = self.arm_list[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)

        return out_feat_list


class PPContextModule(nn.Module):
    """
    Simple Context module.

    Args:
        in_channels (int): The number of input channels to pyramid pooling module.
        inter_channels (int): The number of inter channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
        align_corners (bool): An argument of F.interpolate. It should be set to False
            when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=None):
        super().__init__()

        self.stages = nn.ModuleList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_out = ConvBNReLU(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True)

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = input.shape[2:]
        # print("PPContextModule:")
        # print("original shape:", input_shape)
        for stage in self.stages:
            x = stage(input)
            # print("after pooling shape:",x.shape)
            x = F.interpolate(
                x,
                input_shape, # 1/32 并不小啊，直接 1*1 resize 到 58，40，太粗糙了？最好修改一下这个参数
                mode='nearest',
                align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out += x

        out = self.conv_out(out)
        return out


class SegHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(
            in_chan,
            mid_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        # print("="*100)
        # print("out:",mid_chan, "n_classes:",n_classes)
        self.conv_out = nn.Conv2d(
            mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


# k s p = 3, 2, 1 1*1->3*3
class SpectralExtraction(nn.Module):
    def __init__(self, in_channels, spectral_inter_chs):
        #  spectral_inter_chs 为列表形式，包含5个数，表示5各阶段的通道数量
        super(SpectralExtraction, self).__init__()
        self.spectralExtractionStage = nn.ModuleList()
        assert len(spectral_inter_chs) == 5, "the len of spectral_inter_chs should be 5"
        for i, layer in enumerate(spectral_inter_chs):
            if i == 0:
                per_stage = nn.Sequential(
                    nn.Conv2d(in_channels, layer // 2, kernel_size=1, stride=1, padding=0),
                    nn.LeakyReLU(),
                    nn.Conv2d(layer // 2, layer, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(layer, layer, kernel_size=1, stride=1, padding=0),
                    nn.LeakyReLU()
                )
            else:
                per_stage = nn.Sequential(
                    nn.Conv2d(spectral_inter_chs[i-1], layer // 2, kernel_size=1, stride=1, padding=0),
                    nn.LeakyReLU(),
                    nn.Conv2d(layer // 2, layer, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(layer, layer, kernel_size=1, stride=1, padding=0),
                    nn.LeakyReLU()
                )
            self.spectralExtractionStage.append(per_stage)

    def forward(self, x):
        feature_list = []
        for per_stage in self.spectralExtractionStage:
            x = per_stage(x)
            feature_list.append(x)
        return feature_list


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 5), 'kernel size must be 3 or 5'
        padding = 2 if kernel_size == 5 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PPLiteAddSpectralSeg(nn.Module):

    def __init__(self,
                 num_classes = 3,
                 # backbone = STDC2(),
                 input_channel=3,
                 spectral_input_channels=12,
                 spectral_inter_chs=[18, 24, 32, 64, 96],
                 backbone_indices=[2, 3, 4],
                 arm_type='UAFM_SpAtten',
                 cm_bin_sizes=[4, 8, 16],
                 cm_out_ch=128,
                 # arm_out_chs=[64, 96, 128], #模型太大训练不起来
                 arm_out_chs=[32, 64, 96],
                 # seg_head_inter_chs=[64, 64, 64],
                 seg_head_inter_chs=[32, 32, 32],
                 resize_mode='nearest',
                 pretrained=False):
        super().__init__()

        backbone = STDC2(input_channel)
        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should has feat_channels."
        assert len(backbone.feat_channels) >= len(backbone_indices), \
            f"The length of input backbone_indices ({len(backbone_indices)}) should not be" \
            f"greater than the length of feat_channels ({len(backbone.feat_channels)})."
        assert len(backbone.feat_channels) > max(backbone_indices), \
            f"The max value ({max(backbone_indices)}) of backbone_indices should be " \
            f"less than the length of feat_channels ({len(backbone.feat_channels)})."
        self.backbone = backbone
        self.spectral_backbone = SpectralExtraction(spectral_input_channels, spectral_inter_chs)
        self.spectralAttention_blocks = nn.ModuleList()

        assert len(backbone_indices) > 1, "The lenght of backbone_indices " \
                                          "should be greater than 1"
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]
        # 通道注意力
        for i in range(len(self.backbone_indices)):
            if i == 0:
                self.spectralAttention_blocks.append(SpatialAttention(5))
            else:
                self.spectralAttention_blocks.append(SpatialAttention(3))

        # head
        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
        assert len(arm_out_chs) == len(backbone_indices), "The length of " \
                                                          "arm_out_chs and backbone_indices should be equal"

        self.ppseg_head = PPLiteSegHead(backbone_out_chs, arm_out_chs,
                                        cm_bin_sizes, cm_out_ch, arm_type,
                                        resize_mode)

        if len(seg_head_inter_chs) == 1:
            seg_head_inter_chs = seg_head_inter_chs * len(backbone_indices)
        assert len(seg_head_inter_chs) == len(backbone_indices), "The length of " \
                                                                 "seg_head_inter_chs and backbone_indices should be equal"
        self.seg_heads = nn.ModuleList()  # [..., head_16, head32]
        print("arm_out_chs:",arm_out_chs, " ; seg_head_inter_chs:",seg_head_inter_chs)
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, num_classes))  # to adjust feature

        # pretrained
        self.pretrained = pretrained
        # self.init_weight()

    def forward(self, x, y):
        x_hw = x.shape[2:]
        # print("x_hw:",x_hw)

        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        spectral_feats_backbone = self.spectral_backbone(y)
        # print(type(feats_backbone))
        assert len(feats_backbone) >= len(self.backbone_indices), \
            f"The nums of backbone feats ({len(feats_backbone)}) should be greater or " \
            f"equal than the nums of backbone_indices ({len(self.backbone_indices)})"

        feats_selected = [feats_backbone[i] for i in self.backbone_indices]
        spectral_feats_selected = [spectral_feats_backbone[i] for i in self.backbone_indices]
        # print("spectral_feats_selected")
        # for fea in spectral_feats_selected:
        #     print(fea.shape)
        # 在这里进行融合，然后在使用 ppseg 融合
        spectral_att = [spectralAttention_conv(spectral_feats_selected[i]) for i, spectralAttention_conv in
                       enumerate(self.spectralAttention_blocks)]

        feats_add_spectral = [feats_selected[i] * spectral_att[i] for i in range(len(self.backbone_indices))]

        feats_head = self.ppseg_head(feats_add_spectral)  # [..., x8, x16, x32]
        # 看看feats_head的尺寸就好，然后在这里融合, 应该在 ppseg_head 融合之前融合
        # for fea in feats_head:
        #     print(fea.shape)

        if self.training:
            logit_list = []

            for x, seg_head in zip(feats_head, self.seg_heads):
                x = seg_head(x)
                logit_list.append(x)

            logit_list = [
                F.interpolate(
                    x, x_hw, mode='bilinear', align_corners=None)
                for x in logit_list
            ]
        else:
            x = self.seg_heads[0](feats_head[0])
            # print("x:",x.shape)
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=None)
            logit_list = [x]

        return logit_list

    # def init_weight(self):
    #     if self.pretrained is not None:
    #         utils.load_entire_model(self, self.pretrained)


class PPLiteRgbCatSpectral(nn.Module):

    def __init__(self,
                 num_classes = 3,
                 # backbone = STDC2(),
                 input_channel=3,
                 spectral_input_channels=12,
                 spectral_inter_chs=[18, 24, 32, 64, 96],
                 backbone_indices=[2, 3, 4],
                 arm_type='UAFM_SpAtten',
                 spectral_arm_type='UAFM_ChAtten',
                 cm_bin_sizes=[4, 8, 16],
                 cm_out_ch=128,
                 # arm_out_chs=[64, 96, 128], #模型太大训练不起来
                 arm_out_chs=[32, 64, 96],
                 # seg_head_inter_chs=[64, 64, 64],
                 seg_head_inter_chs=[32, 32, 32],
                 resize_mode='nearest',
                 pretrained=False):
        super().__init__()

        backbone = STDC2(input_channel)
        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should has feat_channels."
        assert len(backbone.feat_channels) >= len(backbone_indices), \
            f"The length of input backbone_indices ({len(backbone_indices)}) should not be" \
            f"greater than the length of feat_channels ({len(backbone.feat_channels)})."
        assert len(backbone.feat_channels) > max(backbone_indices), \
            f"The max value ({max(backbone_indices)}) of backbone_indices should be " \
            f"less than the length of feat_channels ({len(backbone.feat_channels)})."
        self.backbone = backbone
        self.spectral_backbone = SpectralExtraction(spectral_input_channels, spectral_inter_chs)
        self.spectralAttention_blocks = nn.ModuleList()

        assert len(backbone_indices) > 1, "The lenght of backbone_indices " \
                                          "should be greater than 1"
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]
        spectral_backbone_out_chs = [spectral_inter_chs[i] for i in backbone_indices]
        # 通道注意力
        for i in range(len(self.backbone_indices)):
            if i == 0:
                self.spectralAttention_blocks.append(SpatialAttention(5))
            else:
                self.spectralAttention_blocks.append(SpatialAttention(3))

        # head
        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
        assert len(arm_out_chs) == len(backbone_indices), "The length of " \
                                                          "arm_out_chs and backbone_indices should be equal"

        self.ppseg_head = PPLiteSegHead(backbone_out_chs, arm_out_chs,
                                        cm_bin_sizes, cm_out_ch, arm_type,
                                        resize_mode)

        self.spectral_seg_head = PPLiteSegHead(spectral_backbone_out_chs, arm_out_chs,
                                        cm_bin_sizes, cm_out_ch, spectral_arm_type,
                                        resize_mode)

        if len(seg_head_inter_chs) == 1:
            seg_head_inter_chs = seg_head_inter_chs * len(backbone_indices)
        assert len(seg_head_inter_chs) == len(backbone_indices), "The length of " \
                                                                 "seg_head_inter_chs and backbone_indices should be equal"
        self.seg_heads = nn.ModuleList()  # [..., head_16, head32]
        print("arm_out_chs:",arm_out_chs, " ; seg_head_inter_chs:",seg_head_inter_chs)
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs):
            self.seg_heads.append(SegHead(in_ch * 2, mid_ch, num_classes))  # to adjust feature

        # pretrained
        self.pretrained = pretrained
        # self.init_weight()

    def forward(self, x, y):
        x_hw = x.shape[2:]
        # print("x_hw:",x_hw)

        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        spectral_feats_backbone = self.spectral_backbone(y)
        # print(type(feats_backbone))
        assert len(feats_backbone) >= len(self.backbone_indices), \
            f"The nums of backbone feats ({len(feats_backbone)}) should be greater or " \
            f"equal than the nums of backbone_indices ({len(self.backbone_indices)})"

        feats_selected = [feats_backbone[i] for i in self.backbone_indices]
        spectral_feats_selected = [spectral_feats_backbone[i] for i in self.backbone_indices]
        # print("spectral_feats_selected")
        # for fea in spectral_feats_selected:
        #     print(fea.shape)
        # print("feats_selected")
        # for fea in feats_selected:
        #     print(fea.shape)
        # 在这里进行融合，然后在使用 ppseg 融合
        # 如果使用注意力的话，最好也参加 loss 的计算！！！
        # spectral_att = [spectralAttention_conv(spectral_feats_selected[i]) for i, spectralAttention_conv in
        #                enumerate(self.spectralAttention_blocks)]

        # feats_add_spectral = [feats_selected[i] * spectral_att[i] for i in range(len(self.backbone_indices))]

        feats_head = self.ppseg_head(feats_selected)  # [..., x8, x16, x32]
        spectral_feats_head = self.spectral_seg_head(spectral_feats_selected)  # [..., x8, x16, x32]
        feats_head_add_spectral = [torch.cat((feats_head_single, spectral_feats_head_single), dim=1) for
                                   feats_head_single,spectral_feats_head_single in
                                   zip(feats_head, spectral_feats_head)]
        # 看看feats_head的尺寸就好，然后在这里融合, 应该在 ppseg_head 融合之前融合
        # for fea in feats_head:
        #     print(fea.shape)

        if self.training:
            logit_list = []

            # for x, seg_head in zip(feats_head, self.seg_heads):
            for x, seg_head in zip(feats_head_add_spectral, self.seg_heads):
                x = seg_head(x)
                logit_list.append(x)

            logit_list = [
                F.interpolate(
                    x, x_hw, mode='bilinear', align_corners=None)
                for x in logit_list
            ]
        else:
            x = self.seg_heads[0](feats_head_add_spectral[0])
            # print("x:",x.shape)
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=None)
            logit_list = [x]

        return logit_list

    # def init_weight(self):
    #     if self.pretrained is not None:
    #         utils.load_entire_model(self, self.pretrained)

# 
# def get_seg_model(**kwargs):
#     model = PPLiteSeg(pretrained=False)
#     return model
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings("ignore")
# from torch.cuda.amp import autocast, GradScaler
if __name__ == "__main__":
    cm_bin_sizes = [4, 8, 16]
    spectral_inter_chs = [24, 32, 64, 96, 128]
    # model = PPLiteSeg(num_classes=3, input_channel=3, cm_bin_sizes=cm_bin_sizes).cuda() # PPLiteAddSpectralSeg
    model = PPLiteRgbCatSpectral(num_classes=3, input_channel=3, spectral_input_channels=12,
                                 cm_bin_sizes=cm_bin_sizes, spectral_inter_chs=spectral_inter_chs).cuda()
    # model.train()
    model.eval()
    x = torch.randn(2, 3, 1415, 1859).cuda()
    y = torch.randn(2, 12, 1415, 1859).cuda()
    import numpy as np
    label = np.random.randint(0,3,(2, 1415, 1859))
    label = torch.from_numpy(label).cuda()
    with torch.no_grad():
        output = model(x, y)
    print(len(output))
    print(output[0].shape)
    # output = model(x)
    from criterion import OhemCrossEntropy
    criterion = OhemCrossEntropy(ignore_label=0,
                                 thres=0.9,
                                 min_kept=131072,
                                 weight= None,
                                 model_num_outputs=3,
                                 loss_balance_weights = [1,1,1])
    # torch.save(model.state_dict(), 'cat.pth')
    # print(model)
    # print(len(y))
    # for i in range(len(y)):
    #     print(y[i].shape)

    # spect_data = torch.randn(2, 12, 1415, 1859).cuda()
    # feats = model2(spect_data)
    # for feat in feats:
    #     print(feat.shape)
    # losses = criterion(output, label)
    # torch.unsqueeze(losses, 0)
    # print(losses.shape)
    # loss = losses.mean()
    # # print(loss.shape)
    # scaler = GradScaler()
    # scaler.scale(loss).backward()
