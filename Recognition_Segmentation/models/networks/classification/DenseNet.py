import sys
from torch import nn
import torchvision.models as models
sys.path.append('../')
from models.networks.network import Net
import math

class DenseNet(Net):

    def __init__(self, cf, num_classes=21, pretrained=False, net_name='densenet'):
        super(DenseNet, self).__init__(cf)
        self.pretrained = pretrained
        self.net_name = net_name

        self.model = models.densenet121(pretrained=False)
        self.model.classifier = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        print("hey good news, your're actually using densenet, here's your net bro:")
        print(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        pass