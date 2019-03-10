import sys
from torch import nn
import torchvision.models.vgg as models
sys.path.append('../')
from models.networks.network import Net
import math

class OurNet(Net):
    """
    As implemented in /home/grupo09/M3/week5/comp_models.py
    """
    def __init__(self, cf, num_classes=21, pretrained=False, net_name='ournet'):
        super(OurNet, self).__init__(cf)

        self.pretrained = pretrained
        self.net_name = net_name

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(2, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d((3, 3), stride=(2, 2)),
            # depthwise separable conv2d, see:
            # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
            nn.Conv2d(64, 64, kernel_size=(3, 3), groups=64),
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d((3, 3), stride=(2, 2)),
            nn.Conv2d(128, 128, kernel_size=(3, 3), groups=128),
            nn.Conv2d(128, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d((3, 3), stride=(2, 2)),
            nn.Conv2d(256, 128, (1, 1), stride=(2, 2), bias=False))

        self.fc = nn.Sequential(
            # execute the sequential conv in a python console and forward a random tensor, the out shape gives the size of the output feature maps
            nn.Linear(128*6*6, 128),
            nn.ReLU(),
            # apparently we don't need a softmax act., pytorch just does a max(output) when doint the xentropy
            nn.Linear(128, 8)  
            )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)  # flatten
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
