from pdb import set_trace as T

import torch as t
import torchvision
import torch.nn as nn

def ResNetFeatureExtractor():
   resnet = torchvision.models.resnet101(pretrained=True)
   return nn.Sequential(
         resnet.conv1,
         resnet.bn1, 
         resnet.relu,
         resnet.maxpool,
         resnet.layer1,
         resnet.layer2,
         resnet.layer3).eval()
