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


class ResNetflow(nn.Module):
    __factory = {
        18: resnet18,
        34: resnet34,
        50: resnet50,
        101: resnet101,
        152: resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_classes=0, num_features=0, norm=False, dropout=0):
        super(ResNetflow, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        # Construct base (pretrained) resnet
        if depth not in ResNetflow.__factory:
            raise KeyError("Unsupported depth:", depth)

        ### At the bottom of the CNN network
        old_inputweights = ResNetflow.__factory[depth](pretrained=pretrained).conv1.weight.data
        conv0 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init.kaiming_normal(conv0.weight, mode='fan_out')
        add_inputweights = conv0.weight.data
        new_inputweights = tensorcat([old_inputweights, add_inputweights], 1)
        conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1.weight = Parameter(new_inputweights)
        self.conv1 = conv1
        self.base = ResNetflow.__factory[depth](pretrained=pretrained)


        if not self.cut_at_pooling:

            self.num_classes = num_classes
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0

            out_planes = self.base.fc.in_features

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


    def forward(self, x):

        x = self.conv1(x)
        for name, module in self.base._modules.items():
            if name == 'conv1':
                continue
            if name == 'avgpool':
                break
            x = module(x)

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
        for m in self.modules():
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






