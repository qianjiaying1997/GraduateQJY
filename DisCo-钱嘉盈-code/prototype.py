#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:29:12 2021

@author: a
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name,num_classes = 10,num_hidden_units=512,s=2):
        super(VGG, self).__init__()
        self.scale=s
        self.features = self._make_layers(cfg[vgg_name])
        self.preluip1 = nn.PReLU()
        self.dce=dce_loss(num_classes,num_hidden_units)

    def forward(self, x):
        out = self.features(x)
        fea = out.view(out.size(0), -1)
        x1 = self.preluip1((fea))
        centers,x=self.dce(x1)
        output = F.log_softmax(self.scale*x, dim=1)
        return x1,centers,x,output

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class ProNet_mnist(nn.Module):
    def __init__(self, channel_in = 1,num_hidden_units=32, num_classes=10,s=1):
        super(ProNet_mnist, self).__init__()
        self.scale=s
        self.conv1_1 = nn.Conv2d(channel_in, 16, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
#        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
#        self.prelu3_1 = nn.PReLU()
#        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
#        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(32 * 7 * 7, num_hidden_units)
        self.dce=dce_loss_mean(num_classes,num_hidden_units)


    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
#        x = self.prelu3_1(self.conv3_1(x))
#        x = self.prelu3_2(self.conv3_2(x))
#        x = F.max_pool2d(x, 2)
        x= x.view(-1, 32 * 7 * 7)
        x1 = self.preluip1(self.ip1(x))

        centers,x=self.dce(x1)
        output = F.softmax(x/self.scale, dim=1)
        return x1, centers,x,output


class ProNet_MNIST(nn.Module):
    def __init__(self, num_classes=10, num_hidden_units=500, s=1):
        super(ProNet_MNIST, self).__init__()
        self.scale = s
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, num_hidden_units)
        self.dce = dce_loss_mean(num_classes,num_hidden_units)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        fea = F.relu(self.fc1(x))

        centers,x = self.dce(fea)
        output = F.softmax(x/self.scale, dim=1)
        return fea,centers,x,output



class ProNet_mean(nn.Module):
    def __init__(self, channel_in = 3,num_hidden_units=16, num_classes=10,s=2):
        super(ProNet_mean, self).__init__()
        self.scale=s
        self.conv1_1 = nn.Conv2d(channel_in, 128, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
#        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
#        self.prelu3_1 = nn.PReLU()
#        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
#        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
#        self.ip1 = nn.Linear(256 * 4 * 4, num_hidden_units)
        self.ip1 = nn.Linear(256 * 8 * 8, num_hidden_units)
        self.dce=dce_loss_mean(num_classes,num_hidden_units)


    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
#        x = self.prelu3_1(self.conv3_1(x))
#        x = self.prelu3_2(self.conv3_2(x))
#        x = F.max_pool2d(x, 2)
        x= x.view(-1, 256 * 8 * 8)
        x1 = self.preluip1(self.ip1(x))

        centers,x=self.dce(x1)
        output = F.softmax(x/self.scale, dim=1)
        return x1, centers,x,output

class ProNet(nn.Module):
    def __init__(self, channel_in = 3,num_hidden_units=16, num_classes=10,s=0.1):
        super(ProNet, self).__init__()
        self.scale=s
        self.fc1 = nn.Sequential(
            nn.Linear(64, num_hidden_units),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.BatchNorm1d(num_hidden_units))
        self.dce=dce_loss_mean(num_classes,num_hidden_units)


    def forward(self, x):
        x1=self.fc1(x)
        centers ,x=self.dce(x1)
        output = F.softmax(x/self.scale, dim=1)
        return x1, centers,  x,output


class dce_loss_mean(torch.nn.Module):
    def __init__(self, n_classes,feat_dim,init_weight=True):

        super(dce_loss_mean, self).__init__()
        self.n_classes=n_classes
        self.feat_dim=feat_dim
        self.centers=nn.Parameter(torch.randn(self.n_classes,self.feat_dim).cuda(),requires_grad=True)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)



    def forward(self, x):

        features_square=torch.sum(torch.pow(x,2),1, keepdim=True)
        centers_square=torch.sum(torch.pow(self.centers.t(),2),0, keepdim=True)
        features_into_centers=2*torch.matmul(x, (self.centers.t()))
        dist=features_square+centers_square-features_into_centers


        return self.centers, -dist

class dce_loss(torch.nn.Module):
    def __init__(self, n_classes,feat_dim,init_weight=True):

        super(dce_loss, self).__init__()
        self.n_classes=n_classes
        self.feat_dim=feat_dim
        self.centers=nn.Parameter(torch.randn(self.n_classes,self.feat_dim).cuda(),requires_grad=True)
        self.sigma = nn.Parameter(torch.randn(self.n_classes,self.feat_dim).cuda(),requires_grad=True)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)
#        nn.init.kaiming_normal(self.sigma)
        nn.init.ones_(self.sigma)


    def forward(self, x):
        dist = torch.zeros([x.shape[0],self.n_classes]).cuda()
        for i in range(self.n_classes):
            A = torch.diag(self.sigma[i])
            m = x-self.centers[i]
            M = torch.mm(m,torch.mm(A,m.t()))
            dist[:,i] = torch.diag(M)

#        print('dist shape',dist.shape)
        return self.centers, self.sigma, -dist

def regularization(features, centers, labels):
#        distance=(features-torch.t(centers)[labels])
        distance = (features-(centers)[labels])
        distance=torch.sum(torch.pow(distance,2),1, keepdim=True)
#        distance = torch.sum(torch.abs(distance),1, keepdim=True)
        distance = torch.sum(distance,0,keepdim=True)
        distance = distance/features.shape[0]

#        distance=(torch.sum(distance, 0, keepdim=True))/features.shape[0]

        return distance