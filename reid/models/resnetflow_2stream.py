from __future__ import absolute_import

import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch import cat as tensorcat
from ..utils import *
from ..utils.visualize import*
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, \
    resnet152


class ResNetflow2stream(nn.Module):
    __factory = {
        18: resnet18,
        34: resnet34,
        50: resnet50,
        101: resnet101,
        152: resnet152,
    }


    def __init__(self, depth1, depth2, pretrained=True, cut_at_pooling=False,
                 num_classes=0, num_features=0, norm=False, dropout=0):
        super(ResNetflow2stream, self).__init__()

        self.depth1 = depth1
        self.depth2 = depth2
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        # Construct base (pretrained) resnet
        if depth1 not in ResNetflow2stream.__factory:
            raise KeyError("Unsupported depth1:", depth1)

        if depth2 not in ResNetflow2stream.__factory:
            raise KeyError("Unsupported depth2:", depth2)

        ### At the bottom of the CNN network

        ### stream1 for the image
        self.base1 = ResNetflow2stream.__factory[depth1](pretrained=True)

        #### stream2 for the opticalflow
        conv1_base2 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init.kaiming_normal(conv1_base2.weight, mode='fan_out')
        self.conv1_base2 = conv1_base2
        self.base2 = ResNetflow2stream.__factory[depth2](pretrained=pretrained)

        #### fusion layers
        out_planes = self.base1.fc.in_features
        print(out_planes)
        print(self.base2.fc.in_features)
        self.conv_fusion = nn.Conv2d(out_planes*2, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(out_planes)




        ### At the top of the CNN network
        # Append new layers

        if not self.cut_at_pooling:

            self.num_classes = num_classes
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0

            out_planes = self.base1.fc.in_features
            print(out_planes)
            print(self.bases.fc.in_features)

            ### At the top of the CNN network
            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)

                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)

            else:

                # Change the num_features to CNN output channels
                self.num_features = out_planes

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)

            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)


        if not self.pretrained:
            self.reset_params()


    def forward(self, input):
        x_img = input[:, 0:3 ,...]
        x_optical = input[:, 3:5, ...]

        for name, module in self.base1._modules.items():

            if name == 'avgpool':
                break
            x_img = module(x_img)

        x_optical = self.conv1_base2(x_optical)

        for name, module in self.base2._modules.items():
            if name == 'conv1':
                continue
            if name == 'avgpool':
                break
            x_optical = module(x_optical)

        #%%%%%%%%%%% fusion convolution %%%%%%%%%%%%
        x_fusion = torch.cat([x_img, x_optical], 1)
        x_fusion = self.conv_fusion(x_fusion)
        x = self.bn_fusion(x_fusion)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = x / x.norm(2, 1).expand_as(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x







    def reset_params(self):
        for m in self.base2.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)
