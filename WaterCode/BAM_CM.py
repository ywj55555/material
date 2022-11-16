import torch
from torch import nn

class BAM_CM(nn.Module):#No Attention, three losses
    def __init__(self, band, classes):
        super(BAM_CM, self).__init__()
        BAM1 = [
            nn.Conv2d(in_channels=band, out_channels=16, padding=1, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        ]
        BAM2 = [
            nn.Conv2d(in_channels=16, out_channels=32, padding=1, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=32, out_channels=32, padding=1, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        ]
        BAM3 = [
            nn.Conv2d(in_channels=32, out_channels=32, padding=1, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=32, out_channels=32, padding=1, kernel_size=3, stride=1),
            nn.AvgPool2d(kernel_size=(11,11),padding=5,stride=1),
            nn.Conv2d(in_channels=32, out_channels=60, padding=0, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=60, out_channels=band, padding=0, kernel_size=1, stride=1),
            nn.Sigmoid()
        ]
        self.BAM1 = nn.Sequential(*BAM1)
        self.BAM2 = nn.Sequential(*BAM2)
        self.BAM3 = nn.Sequential(*BAM3)

        VGG = [
            nn.Conv2d(in_channels=band, out_channels=64, padding=1, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels=512, out_channels=4096, padding=1, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=4096, out_channels=4096, padding=1, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=4096, out_channels=classes, padding=1, kernel_size=1, stride=1),
        ]
        self.VGG = nn.Sequential(*VGG)
    def forward(self, X):
        X_B1 = self.BAM1(X)
        X_B2 = self.BAM2(X_B1)
        X_B3 = self.BAM3(X_B2)
        X = X * X_B3
        X_out = self.VGG(X)
        return X_out

if __name__ == "__main__":
    dummy_input = torch.rand(1, 128, 259, 512).cuda()
    model = BAM_CM(128,31).cuda()
    predict = model(dummy_input)
    print(predict.shape)