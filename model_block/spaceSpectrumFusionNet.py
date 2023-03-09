import torch.nn as nn
import numpy as np
import torch

class DepthSepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=0, bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1):
        super(DepthSepConv, self).__init__()

        self.conv_dw = nn.Conv2d(
            int(in_channels * channel_multiplier), int(in_channels * channel_multiplier), kernel_size,
            stride=stride, groups=int(in_channels * channel_multiplier), dilation=dilation, padding=padding)

        self.conv_pw = nn.Conv2d(int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding,
                                 bias=bias)

        self.relu = nn.ReLU(inplace=True)

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        x = self.relu(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, ratio=1.0, version=2, pool_size=3):
        super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, int(channel * ratio), kernel_size=1, stride=1),
            # nn.LeakyReLU(),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(channel * ratio), channel, kernel_size=1, stride=1),
        )
        self.version = version
        self.pool_padding = int(pool_size / 2)
        self.average_pool = nn.AvgPool2d(kernel_size=pool_size, stride=1, padding=self.pool_padding, ceil_mode=False,
                                         count_include_pad=False)  # 一定不能包含0填充！！
    def forward(self, x):
        # assert x.size(2) == 1 and x.size(3) == 1
        y = self.conv1(x)
        y2 = torch.sigmoid(self.conv2(y))  # 对所有值都进行映射
        # 要不要加一个 池化平滑一下？？？ 3.0版本才平滑
        if self.version == 3 and self.pool_padding > 0:
            y2 = self.average_pool(y2)
        return x * y2

class weightingLayer(nn.Module):
    def __init__(self, channel, ratio=1):
        super(weightingLayer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, int(channel * ratio), kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(channel * ratio), 1, kernel_size=1, stride=1),
        )
    def forward(self, x):
        y = self.conv1(x)
        y2 = torch.sigmoid(self.conv2(y))  # 需要在通道方向进行广播，相乘会自动广播！！
        return x * y2

def repeat_spectrum(input_num, output_num, conv_size=1, pool_size=3):
    layers = [
        nn.Sequential(
            nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=0),  # stride 不能用默认值
            nn.Conv2d(input_num, output_num, kernel_size=conv_size, stride=1, padding=0),
            nn.BatchNorm2d(output_num),
            nn.LeakyReLU()
        )
        ]
    return nn.Sequential(*layers)

def repeat_space(input_num, output_num, conv_size=3):
    layers = [
        nn.Sequential(
            nn.Conv2d(input_num, output_num, kernel_size=conv_size, stride=1, padding=0),
            nn.BatchNorm2d(output_num),
            nn.LeakyReLU()
        )
        ]
    return nn.Sequential(*layers)

class spaceSpectrumFusionNet(nn.Module):
    def __init__(self, bands=9, class_num=4, ratio=1.8, patch_size=11, spectrum_used=True, space_used=True, fusion=True,
                 version=2):
        super(spaceSpectrumFusionNet, self).__init__()
        # 谱支路 1*1 然后使用 max 池化
        self.spectrum_used = spectrum_used
        self.space_used = space_used
        self.fusion = fusion
        self.patch_size = patch_size
        self.inner_channel = int(bands * ratio)
        self.inner_channel_later = int(bands * ratio * 0.6)
        self.blocks_kernel_size = [3, 4, 4, 3]
        self.fusion_num = 3
        self.version = version
        print('self.spectrum_used ', self.spectrum_used )
        print('self.space_used ', self.space_used)
        print('self.patch_size ', self.patch_size)
        print('self.version ', self.version)
        if not self.spectrum_used or not self.space_used:
            self.fusion_num = 1
        if self.patch_size == 9:
            self.blocks_kernel_size = [3, 3, 3, 3]
        elif self.patch_size == 7:
            self.blocks_kernel_size = [3, 1, 3, 3]
        elif self.patch_size == 5:
            self.blocks_kernel_size = [3, 1, 1, 3]


        self.spectrum_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bands, bands, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(bands),
                nn.LeakyReLU()
            ),

            SELayer(bands, 1, self.version, self.blocks_kernel_size[0]),  # 通道注意力
            repeat_spectrum(bands, self.inner_channel, 1, self.blocks_kernel_size[0]),
            nn.Identity(),

            SELayer(self.inner_channel, 1, self.version, self.blocks_kernel_size[0]),  # 通道注意力
            repeat_spectrum(self.inner_channel, self.inner_channel, 1, self.blocks_kernel_size[1]),
            nn.Identity(),

            SELayer(self.inner_channel, 1, self.version, self.blocks_kernel_size[0]),  # 通道注意力
            repeat_spectrum(self.inner_channel, self.inner_channel_later, 1, self.blocks_kernel_size[2]),
            nn.Identity(),

            SELayer(self.inner_channel_later, 1.2, self.version, self.blocks_kernel_size[0]),  # 通道注意力
            repeat_spectrum(self.inner_channel_later, class_num, 1, self.blocks_kernel_size[3]),
            nn.Identity(),  # 1 * 1
        ])
        # self.spectrum_attention = SELayer(class_num, 2)

        # 形支路 主要使用3*3 4*4
        self.space_branch = nn.ModuleList([
            DepthSepConv(bands, self.inner_channel, kernel_size=self.blocks_kernel_size[0]),
            nn.Identity(),
            DepthSepConv(self.inner_channel, self.inner_channel, kernel_size=self.blocks_kernel_size[1]),
            nn.Identity(),
            DepthSepConv(self.inner_channel, self.inner_channel_later, kernel_size=self.blocks_kernel_size[2]),
            nn.Identity(),
            DepthSepConv(self.inner_channel_later, class_num, kernel_size=self.blocks_kernel_size[3]),
            nn.Identity(),  # 1 * 1
        ])

        self.fusion_branch = nn.ModuleList([
            DepthSepConv(self.inner_channel * 2, self.inner_channel, kernel_size=self.blocks_kernel_size[1]),
            # nn.Identity(),
            DepthSepConv(self.inner_channel * 3, self.inner_channel_later, kernel_size=self.blocks_kernel_size[2]),
            # nn.Identity(),
            DepthSepConv(self.inner_channel_later * 3, class_num, kernel_size=self.blocks_kernel_size[3]),
            # nn.Identity(),
        ])

        self.spectrum_weighting = weightingLayer(class_num, 2)
        self.space_weighting = weightingLayer(class_num, 2)
        self.fusion_weighting = weightingLayer(class_num, 2)

        self.cls_pred_conv = nn.Conv2d(class_num * self.fusion_num, class_num, kernel_size=1)

    def forward(self, spec_data, space_data):
        # 谱支路
        # print('spaceSpectrumFusionNet forward')
        spectrum_feat_list = []
        spectrum_predict = None
        if self.spectrum_used:
            for op in self.spectrum_branch:
                spec_data = op(spec_data)
                if isinstance(op, nn.Identity):
                    spectrum_feat_list.append(spec_data)
            spectrum_predict = self.spectrum_weighting(spectrum_feat_list[-1])
        # 形支路
        space_feat_list = []
        space_predict = None
        if self.space_used:
            for op in self.space_branch:
                space_data = op(space_data)
                if isinstance(op, nn.Identity):
                    space_feat_list.append(space_data)
            space_predict = self.space_weighting(space_feat_list[-1])
        # 融合支路
        fusion_feat_list = []
        fusion_predict = None
        final_feat = None
        if self.space_used and self.spectrum_used:
            for spec, spac in zip(spectrum_feat_list[:-1], space_feat_list[:-1]):
                fusion_data = torch.cat((spec, spac), dim=1)
                fusion_feat_list.append(fusion_data)

            for i, feat in enumerate(fusion_feat_list):
                if i != 0:
                    feat = torch.cat((feat, fusion_predict), dim=1)
                fusion_predict = self.fusion_branch[i](feat)
            fusion_predict = self.fusion_weighting(fusion_predict)
            final_feat = torch.cat((fusion_predict, spectrum_predict, space_predict), dim=1)
        elif self.space_used:
            final_feat = space_predict
        elif self.spectrum_used:
            final_feat = spectrum_predict
        logit = self.cls_pred_conv(final_feat)
        return logit

class SpectrumNet(nn.Module):
    def __init__(self, bands=9, class_num=4, ratio=1.8, patch_size=11, version=2):
        super(SpectrumNet, self).__init__()
        # 谱支路 1*1 然后使用 max 池化
        self.patch_size = patch_size
        self.inner_channel = int(bands * ratio)
        self.inner_channel_later = int(bands * ratio * 0.6)
        self.blocks_kernel_size = [3, 4, 4, 3]
        self.version = version
        print('self.patch_size ', self.patch_size)
        print('self.version ', self.version)
        if self.patch_size == 9:
            self.blocks_kernel_size = [3, 3, 3, 3]
        elif self.patch_size == 7:
            self.blocks_kernel_size = [3, 1, 3, 3]
        elif self.patch_size == 5:
            self.blocks_kernel_size = [3, 1, 1, 3]

        self.spectrum_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bands, bands, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(bands),
                nn.LeakyReLU()
            ),
            SELayer(bands, 1, self.version, self.blocks_kernel_size[0]),  # 通道注意力
            repeat_spectrum(bands, self.inner_channel, 1, self.blocks_kernel_size[0]),
            # nn.Identity(),

            SELayer(self.inner_channel, 1, self.version, self.blocks_kernel_size[0]),  # 通道注意力
            repeat_spectrum(self.inner_channel, self.inner_channel, 1, self.blocks_kernel_size[1]),
            # nn.Identity(),

            SELayer(self.inner_channel, 1, self.version, self.blocks_kernel_size[0]),  # 通道注意力
            repeat_spectrum(self.inner_channel, self.inner_channel_later, 1, self.blocks_kernel_size[2]),
            # nn.Identity(),

            SELayer(self.inner_channel_later, 1.2, self.version, self.blocks_kernel_size[0]),  # 通道注意力
            repeat_spectrum(self.inner_channel_later, class_num, 1, self.blocks_kernel_size[3]),
            # nn.Identity(),  # 1 * 1
        ])
        self.cls_pred_conv = nn.Conv2d(class_num, class_num, kernel_size=1)
    def forward(self, spec_data):
        # 谱支路
        for op in self.spectrum_branch:
            spec_data = op(spec_data)
        logit = self.cls_pred_conv(spec_data)
        return logit

class spaceSpectrumFusionNetOld(nn.Module):
    def __init__(self, bands=9, class_num=4, ratio=2, patch_size=11, spectrum_used=True, space_used=True, fusion=True):
        super(spaceSpectrumFusionNetOld, self).__init__()
        # 谱支路 1*1 然后使用 max 池化
        self.spectrum_used = spectrum_used
        self.space_used = space_used
        self.fusion = fusion
        self.patch_size = patch_size
        self.inner_channel = int(bands * ratio)
        self.inner_channel_later = int(bands * ratio * 0.7)
        self.blocks_kernel_size = [3, 4, 4, 3]
        self.fusion_num = 3
        if not self.spectrum_used or not self.space_used:
            self.fusion_num = 1
        if self.patch_size == 9:
            self.blocks_kernel_size = [3, 3, 3, 3]
        elif self.patch_size == 7:
            self.blocks_kernel_size = [3, 1, 3, 3]
        elif self.patch_size == 5:
            self.blocks_kernel_size = [3, 1, 1, 3]


        self.spectrum_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bands, bands, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(bands),
                nn.LeakyReLU()
            ),
            repeat_spectrum(bands, self.inner_channel, 1, self.blocks_kernel_size[0]),
            nn.Identity(),
            repeat_spectrum(self.inner_channel, self.inner_channel, 1, self.blocks_kernel_size[1]),
            nn.Identity(),
            repeat_spectrum(self.inner_channel, self.inner_channel_later, 1, self.blocks_kernel_size[2]),
            nn.Identity(),
            repeat_spectrum(self.inner_channel_later, class_num, 1, self.blocks_kernel_size[3]),
            SELayer(class_num, 2),  # 通道注意力
            nn.Identity(),  # 1 * 1
        ])
        # self.spectrum_attention = SELayer(class_num, 2)

        # 形支路 主要使用3*3 4*4
        self.space_branch = nn.ModuleList([
            DepthSepConv(bands, self.inner_channel, kernel_size=self.blocks_kernel_size[0]),
            nn.Identity(),
            DepthSepConv(self.inner_channel, self.inner_channel, kernel_size=self.blocks_kernel_size[1]),
            nn.Identity(),
            DepthSepConv(self.inner_channel, self.inner_channel_later, kernel_size=self.blocks_kernel_size[2]),
            nn.Identity(),
            DepthSepConv(self.inner_channel_later, class_num, kernel_size=self.blocks_kernel_size[3]),
            nn.Identity(),  # 1 * 1
        ])

        self.fusion_branch = nn.ModuleList([
            DepthSepConv(self.inner_channel * 2, self.inner_channel, kernel_size=self.blocks_kernel_size[1]),
            # nn.Identity(),
            DepthSepConv(self.inner_channel * 3, self.inner_channel_later, kernel_size=self.blocks_kernel_size[2]),
            # nn.Identity(),
            DepthSepConv(self.inner_channel_later * 3, class_num, kernel_size=self.blocks_kernel_size[3]),
            # nn.Identity(),
        ])

        self.spectrum_weighting = weightingLayer(class_num, 2)
        self.space_weighting = weightingLayer(class_num, 2)
        self.fusion_weighting = weightingLayer(class_num, 2)

        self.cls_pred_conv = nn.Conv2d(class_num * self.fusion_num, class_num, kernel_size=1)

    def forward(self, spec_data, space_data):
        # 谱支路
        spectrum_feat_list = []
        spectrum_predict = None
        if self.spectrum_used:
            for op in self.spectrum_branch:
                spec_data = op(spec_data)
                if isinstance(op, nn.Identity):
                    spectrum_feat_list.append(spec_data)
            spectrum_predict = self.spectrum_weighting(spectrum_feat_list[-1])
        # 形支路
        space_feat_list = []
        space_predict = None
        if self.space_used:
            for op in self.space_branch:
                space_data = op(space_data)
                if isinstance(op, nn.Identity):
                    space_feat_list.append(space_data)
            space_predict = self.space_weighting(space_feat_list[-1])
        # 融合支路
        fusion_feat_list = []
        fusion_predict = None
        final_feat = None
        if self.space_used and self.spectrum_used:
            for spec, spac in zip(spectrum_feat_list[:-1], space_feat_list[:-1]):
                fusion_data = torch.cat((spec, spac), dim=1)
                fusion_feat_list.append(fusion_data)

            for i, feat in enumerate(fusion_feat_list):
                if i != 0:
                    feat = torch.cat((feat, fusion_predict), dim=1)
                fusion_predict = self.fusion_branch[i](feat)
            fusion_predict = self.fusion_weighting(fusion_predict)
            final_feat = torch.cat((fusion_predict, spectrum_predict, space_predict), dim=1)
        elif self.space_used:
            final_feat = space_predict
        elif self.spectrum_used:
            final_feat = spectrum_predict
        logit = self.cls_pred_conv(final_feat)
        return logit


if __name__ == '__main__':
    # from model_block.materialNet import MaterialSubModel
    model_select = 4
    input_bands_num = 9
    class_num = 4
    patch_size = 5
    if model_select == 1:
        model = spaceSpectrumFusionNet(input_bands_num, class_num, patch_size=patch_size)
    elif model_select == 2:
        model = spaceSpectrumFusionNet(input_bands_num, class_num, space_used=False)  # Spectrum
    elif model_select == 3:
        model = spaceSpectrumFusionNet(input_bands_num, class_num, spectrum_used=False)  # Spectrum
    if model_select == 4:
        model = spaceSpectrumFusionNet(input_bands_num, class_num, patch_size=patch_size, version=3)
    if model_select == 5:
        model = SpectrumNet(input_bands_num, class_num, patch_size=patch_size, version=3)
    if model_select == 6:
        model = spaceSpectrumFusionNet(input_bands_num, class_num, patch_size=patch_size, spectrum_used=False, version=3).cuda()
    # else:
    #     model = MaterialSubModel(input_bands_num, class_num)

    model_root_path = '/home/cjl/ywj_code/graduationCode/alien_material/model/'
    modelpath = 'SkinClothWater18_twoBranch3.0_5_0.001_64_4_handSelect_22276800'
    # model = SpectrumNet(9, 4, patch_size=11, version=3).cuda()
    modelpath = model_root_path + modelpath + '/' + '1' + '.pkl'
    modelpath2 = '/home/cjl/ywj_code/graduationCode/alien_material/model/SkinClothWater18_onlySpectrumBranch3.0_2_11_0.001_64_4_handSelect_22276800/0.pkl'
    # # if not os.path.exists(tmp_model_path):
    # #     continue
    # checkpoint = torch.load(modelpath2)
    # print(22)
    # for k, v in checkpoint.items():
    #     print(k)

    model.load_state_dict(torch.load(modelpath))
    for name, param in model.named_parameters():
        print(name)
    # labelNp = np.random.randint(0, 4, (2, 11, 11))
    # label = torch.tensor(labelNp).cuda().long()
    # import numpy as np
    # model = FreeNet(9, 4).cuda()
    # labelNp = np.random.randint(0, 4, (2, 1008, 1008))
    # label = torch.tensor(labelNp).cuda().long()
    # torch.backends.cudnn.benchmark = False
    # # 配合随机数种子，确保网络多次训练参数一致
    # torch.backends.cudnn.deterministic = True
    # # 使用非确定性算法
    # torch.backends.cudnn.enabled = True
    # spectralData = torch.rand(4, 9, 11, 11).cuda() # 为什么和batch有关？？
    # spaceData = torch.rand(4, 9, 11, 11).cuda()  # 为什么和batch有关？？
    # predict = model(spectralData)
    # torch.save(model.state_dict(), './Spectrum4.0GPU.pkl')
    # print(predict.size())
    # spectralData = torch.rand(2, 9, 1008, 1008).cuda()  # 为什么和batch有关？？
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = OhemCrossEntropy(ignore_label=-1,
    #                              thres=0.9,
    #                              min_kept=500000,
    #                              weight=None,
    #                              model_num_outputs=1,
    #                              loss_balance_weights=[1])
    # # criterion =
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    # # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    # scaler = GradScaler()
    # model.train()
    # for epoch in range(1000):
    #     with autocast():
    #         predict = model(spectralData)
    #         # print(predict.size())
    #     # losses = F.cross_entropy(predict, label, weight=None,
    #     #                          ignore_index=-1, reduction='none')
    #     losses = criterion(predict, label)
    #     loss = losses.mean()
    #     # loss = criterion(predict, label)  # + L2Loss(model, 1e-4)
    #     print(epoch)
    #     print(loss)
    #     # trainLossTotal += loss
    #     # print("loss = %.5f" % float(loss))
    #     # optimizer.zero_grad()
    #     # loss.backward()
    #     # optimizer.step()
    #
    #     model.zero_grad()
    #     scaler.scale(loss).backward()
    #     # trainLossTotal += loss.item()
    #     scaler.step(optimizer)
    #     scaler.update()
    #
    #     # predict = predict.squeeze()
    #     predictIndex = torch.argmax(predict, dim=1)  # 计算一下准确率和召回率 B*H*W 和label1一样
    #     count_right = torch.sum(predictIndex == label).item()  # 全部训练
    #     count_tot = torch.sum(label != -1).item()
    #     print(count_right / count_tot)