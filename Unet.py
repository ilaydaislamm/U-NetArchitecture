import torch.nn as nn
import torchvision.transforms.functional as F
import torch
from torchinfo import summary

class UNet(nn.Module):
    def __init__(self,
                 in_channels : int = 1,
                 out_classes : int = 2):
        super().__init__()

        self.conv_layer1 = nn.Sequential(
                                    nn.Conv2d(in_channels, 64, kernel_size=3, padding = 0),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, padding = 0),
                                    nn.ReLU()
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer2 = nn.Sequential(
                                    nn.Conv2d(64, 128, kernel_size=3, padding = 0),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, kernel_size=3, padding = 0),
                                    nn.ReLU()
        )
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Sequential(
                                    nn.Conv2d(128, 256, kernel_size=3, padding = 0),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, kernel_size=3, padding = 0),
                                    nn.ReLU(),
        )
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer4 = nn.Sequential(
                                    nn.Conv2d(256, 512, kernel_size=3, padding = 0),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, kernel_size=3, padding = 0),
                                    nn.ReLU()
        )
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding = 0),
                                    nn.ReLU(),
                                    nn.Conv2d(1024, 1024, kernel_size=3, padding = 0),
                                    nn.ReLU()
        )

        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        self.up_conv_layer1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.up_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        self.up_conv_layer2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.up_conv_layer3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.up_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.up_conv_layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.last_layer = nn.Conv2d(64, out_classes, kernel_size=1)

        self.apply(self._init_weights)


    def forward(self, x):

        s1 = self.conv_layer1(x)
        x =  self.max_pool1(s1)

        s2 = self.conv_layer2(x)
        x = self.max_pool2(s2)

        s3 = self.conv_layer3(x)
        x = self.max_pool3(s3)

        s4 = self.conv_layer4(x)
        x = self.max_pool4(s4)

        x = self.conv_layer5(x)

        x = self.up_conv1(x)
        x = torch.cat([F.center_crop(s4,x.shape[2:]),x],dim = 1)
        x = self.up_conv_layer1(x)

        x = self.up_conv2(x)
        x = torch.cat([F.center_crop(s3,x.shape[2:]),x],dim = 1)
        x = self.up_conv_layer2(x)

        x = self.up_conv3(x)
        x = torch.cat([F.center_crop(s2, x.shape[2:]), x], dim=1)
        x = self.up_conv_layer3(x)

        x = self.up_conv4(x)
        x = torch.cat([F.center_crop(s1, x.shape[2:]), x], dim=1)
        x = self.up_conv_layer4(x)

        x = self.last_layer(x)

        return x

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)





