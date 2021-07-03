import numpy as np
import pandas as pd
import os
import math
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Models
class EffNet(nn.Module):
    def __init__(self):
        super(EffNet, self).__init__()
        self.eff = EfficientNet.from_pretrained('efficientnet-b5')#, num_classes=4)
        self.l = nn.Linear(1003, 4)
    def forward(self, x, director):
        x = self.eff(x) #(batch_size, 1000)
        x = torch.cat((x, director), 1)
        x = F.softmax(self.l(x), dim=1)
        return x


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, 4)
        
    def forward(self, x):
        x = self.resnet(x)
        x = F.softmax(self.fc(x.reshape(-1, 2048)), dim=1)
        return x



class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.mobile = models.mobilenet_v2(pretrained=True)
        self.mobile.classifier[-1] = nn.Linear(self.mobile.classifier[-1].in_features, 4)
        
    def forward(self, x):
        x = F.softmax(self.mobile(x), 1)
        return x

       
