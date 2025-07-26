from torch import nn
import torch


class DCEDN(nn.Module):

    def __init__(self, num_channels=3):
        super(DCEDN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=3 // 2, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=5 // 2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(128, 256, kernel_size=7, padding=7 // 2, padding_mode='replicate')


        self.conv4 = nn.Conv2d(256, 256, kernel_size=7, padding=7 // 2, padding_mode='replicate')
        self.conv5 = nn.Conv2d(384, 128, kernel_size=5, padding=5 // 2, padding_mode='replicate')
        self.conv6 = nn.Conv2d(192, 64, kernel_size=3, padding=3 // 2, padding_mode='replicate')
        self.conv7 = nn.Conv2d(64, 3, kernel_size=1, padding=0, padding_mode='replicate')
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x1 = self.pool(x)
        x = self.relu(self.conv2(x1))
        x2 = self.pool(x)
        x = self.relu(self.conv3(x2))

        x = self.relu(self.conv4(x))
        x3 = torch.concat((x, x2), 1)
        x = self.upsampling(x3)
        x = self.relu(self.conv5(x))
        x4 = torch.concat((x, x1), 1)
        x = self.upsampling(x4)
        x = self.relu(self.conv6(x))
        x = self.conv7(x)
        x = self.tanh(x)
        return x

