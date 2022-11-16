import torch
from torch import nn



class Two_CNN(nn.Module):#No Attention, three losses
    def __init__(self, band, classes):
        super(Two_CNN, self).__init__()
        self.conv11 = nn.Conv3d(1,20,kernel_size=(16,1,1), stride=4, padding=0)  #20, (16,1,1)
        self.conv12 = nn.Conv3d(20,20,kernel_size=(16,1,1), stride=4, padding=0)#20, (16,1,1)
        self.pool1 = nn.MaxPool3d(kernel_size=(16,1,1), stride=4)#(5,1,1)

        self.conv21 = nn.Conv2d(in_channels=band, out_channels=30, padding=1,
                                kernel_size=3, stride=1)#30, (3,3)
        self.conv22 = nn.Conv2d(in_channels=30, out_channels=30, padding=1,
                                kernel_size=3, stride=1)#30, (3,3)
        self.pool2 =  nn.MaxPool3d(kernel_size=(3,3), padding=1, stride=1)#(2,2)

        self.conv31 = nn.Conv2d(in_channels=70, out_channels=400, padding=0,
                                kernel_size=1, stride=1)#400, (1,1)
        self.conv32 = nn.Conv2d(in_channels=400, out_channels=classes, padding=0,
                                kernel_size=1, stride=1) # classes, (1,1)

    def forward(self, X):
        batch,band,H,W = X.shape
        #spectral:3c-conv
        X_spec = X.unsqueeze(1)
        X11 = self.conv11(X_spec)
        X12 = self.conv12(X11)
        X1 = self.pool1(X12)
        X1 = X1.view(batch,40,H,W)

        #spatial:2d-conv
        X21 = self.conv21(X)
        X22 = self.conv22(X21)
        X2 = self.pool2(X22)
        X_fn = torch.cat([X1,X2],dim=1)
        X_fn_1 = self.conv31(X_fn)
        X_out = self.conv32(X_fn_1)

        return X_out
