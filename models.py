# -*- coding: utf-8 -*-
"""
Models for U-Net

@author: Yash Gupta
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

def ConvEntity(in_channels, out_channels, kernel_size=3, batch_norm=True, stride=1):
    with_batch_norm = nn.Sequential(
              nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),#, stride=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(),
              nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),#, stride=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU()
            )
    without_batch_norm = nn.Sequential(
              nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),#, stride=1),
              nn.ReLU(),
              nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),#, stride=1),
              nn.ReLU()
            )
    if batch_norm:
        return with_batch_norm
    else:
        return without_batch_norm

def MaxPoolEntity(kernel_size=2, dropout=0.5):
    fwd = nn.Sequential(
              nn.MaxPool2d(kernel_size=kernel_size, ceil_mode=True),
              nn.Dropout2d(dropout)
            )
    return fwd

def UpConvEntity(in_channels,out_channels,kernel_size=3):
    # Conv2DTranspose if execution is quick, Upsample if don't need to learn upscaling parameters
    fwd = nn.Sequential(
#               nn.Upsample(scale_factor=2, mode='bilinear'),
#               nn.Conv2d(in_channels=iniNumCh, out_channels=finNumCh, kernel_size=3, padding=1, stride=1),
              nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),
#               nn.Dropout(0.5),
#               nn.BatchNorm2d(finNumCh),
#               nn.ReLU()
            )
    return fwd

class RecEntity(nn.Module): 
    def __init__(self,t,out_channels, kernel_size=3, batch_norm=True, stride=1):
        super(RecEntity,self).__init__()
        self.t = t
        self.fwd = nn.Sequential(
                  nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),#, stride=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU()
              )
        
    def forward(self,x):
        a = self.fwd(x)
        for i in range(self.t):
            a = self.fwd(x + a)
        return a


def R2Entity(in_channels, out_channels, t=2, kernel_size=3, batch_norm=True, stride=1):
    fwd = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),#, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            RecEntity(t,out_channels,kernel_size=3, batch_norm=True, stride=1),
            RecEntity(t,out_channels,kernel_size=3, batch_norm=True, stride=1)
            )
    
    return fwd
    
            
class U_Net(nn.Module):
    def __init__(self,in_channels=3,out_channels=2,ini_num_features=16,depth=5):
        super(U_Net, self).__init__()
        ## encoding
        nf = ini_num_features
        self.depth = depth
        layer_dict = {}
        layer_dict['enc_conv1'] = ConvEntity(in_channels,nf)
        
        for i in range(1,depth):
            layer_dict['max_pool'+str(i)] = MaxPoolEntity()
            layer_dict['enc_conv'+str(i+1)] = ConvEntity(nf,2*nf)
            nf = 2*nf
        
        ##decoding
        for i in reversed(range(1,depth)):
            layer_dict['up_conv'+str(i+1)] = UpConvEntity(nf,nf//2)
            layer_dict['dec_conv'+str(i)] = ConvEntity(nf,nf//2)
            nf = nf//2
        
        ##final               
        layer_dict['final'] = nn.Conv2d(in_channels=nf,out_channels=out_channels,kernel_size=1)
        
#        self.c1 = ConvEntity(in_channels,nf) # in_channels=3 for rgb images, 1 for grayscale
#        self.m1 = MaxPoolEntity()
#        self.c2 = ConvEntity(nf,2*nf)
#        self.m2 = MaxPoolEntity()
#        self.c3 = ConvEntity(2*nf,4*nf)
#        self.m3 = MaxPoolEntity()
#        self.c4 = ConvEntity(4*nf,8*nf)
#        self.m4 = MaxPoolEntity()
#        self.c5 = ConvEntity(8*nf,16*nf)
#        
#        ## decoding
#        self.u1 = UpConvEntity(16*nf,8*nf)
#        self.c6 = ConvEntity(16*nf,8*nf)
#        self.u2 = UpConvEntity(8*nf,4*nf)
#        self.c7 = ConvEntity(8*nf,4*nf)
#        self.u3 = UpConvEntity(4*nf,2*nf)
#        self.c8 = ConvEntity(4*nf,2*nf)
#        self.u4 = UpConvEntity(2*nf,nf)
#        self.c9 = ConvEntity(2*nf,nf)
#        
#        ## final
#        self.fin = nn.Conv2d(in_channels=nf,out_channels=2,kernel_size=1)
        self.layers = layer_dict
        for k in self.layers.keys():
            self.add_module(k,self.layers[k])
        
    def forward(self, x):
        
        a = {}
        a['enc0'] = x
        a['enc1'] = self.layers['enc_conv1'](a['enc0'])
        
        for i in range(1,self.depth):
            temp = self.layers['max_pool'+str(i)](a['enc'+str(i)])
            a['enc'+str(i+1)] = self.layers['enc_conv'+str(i+1)](temp)
            
        a['dec'+str(self.depth)] = a['enc'+str(self.depth)]
        
        for i in reversed(range(1,self.depth)):
            temp1 = self.layers['up_conv'+str(i+1)](a['dec'+str(i+1)])
            temp2 = nn.Sequential(nn.Dropout(0.5))(torch.cat((temp1,a['enc'+str(i)]),dim=1))
            a['dec'+str(i)] = self.layers['dec_conv'+str(i)](temp2)
        
        a['fin'] = self.layers['final'](a['dec1'])
        
#        t1 = self.c1(x)
#        t2 = self.m1(t1)
#        t3 = self.c2(t2)
#        t4 = self.m2(t3)
#        t5 = self.c3(t4)
#        t6 = self.m3(t5)
#        t7 = self.c4(t6)
#        t8 = self.m4(t7)
#        t9 = self.c5(t8)
#                             
#        t10 = self.u1(t9)
#        t11 = nn.Sequential(nn.Dropout(0.5))(torch.cat((t10, t7), dim=1))
#        t12 = self.c6(t11)
#                             
#        t13 = self.u2(t12)
#        t14 = nn.Sequential(nn.Dropout(0.5))(torch.cat((t13, t5), dim=1))
#        t15 = self.c7(t14)
#                             
#        t16 = self.u3(t15)
#        t17 = nn.Sequential(nn.Dropout(0.5))(torch.cat((t16, t3), dim=1))
#        t18 = self.c8(t17)
#                             
#        t19 = self.u4(t18)
#        t20 = nn.Sequential(nn.Dropout(0.5))(torch.cat((t19, t1), dim=1))
#        t21 = self.c9(t20)
#        
#        t22 = self.fin(t21)
        return F.softmax(a['fin'], dim=1)
    
    def describe(self):
        for k,v in self.layers.items():
#            print(k)
            print(k,v)


class R2U_Net(nn.Module):
    def __init__(self,in_channels=3,out_channels=2,ini_num_features=16,depth=5):
        super(R2U_Net, self).__init__()
        ## encoding
        nf = ini_num_features
        self.depth = depth
        layer_dict = {}
        layer_dict['enc_conv1'] = R2Entity(in_channels,nf)
        
        for i in range(1,depth):
            layer_dict['max_pool'+str(i)] = MaxPoolEntity()
            layer_dict['enc_conv'+str(i+1)] = R2Entity(nf,2*nf)
            nf = 2*nf
        
        ##decoding
        for i in reversed(range(1,depth)):
            layer_dict['up_conv'+str(i+1)] = UpConvEntity(nf,nf//2)
            layer_dict['dec_conv'+str(i)] = R2Entity(nf,nf//2)
            nf = nf//2
        
        ##final               
        layer_dict['final'] = nn.Conv2d(in_channels=nf,out_channels=out_channels,kernel_size=1)

        self.layers = layer_dict
        for k in self.layers.keys():
            self.add_module(k,self.layers[k])
        
    def forward(self, x):
        
        a = {}
        a['enc0'] = x
        a['enc1'] = self.layers['enc_conv1'](a['enc0'])
        
        for i in range(1,self.depth):
            temp = self.layers['max_pool'+str(i)](a['enc'+str(i)])
            a['enc'+str(i+1)] = self.layers['enc_conv'+str(i+1)](temp)
            
        a['dec'+str(self.depth)] = a['enc'+str(self.depth)]
        
        for i in reversed(range(1,self.depth)):
            temp1 = self.layers['up_conv'+str(i+1)](a['dec'+str(i+1)])
            temp2 = nn.Sequential(nn.Dropout(0.5))(torch.cat((temp1,a['enc'+str(i)]),dim=1))
            a['dec'+str(i)] = self.layers['dec_conv'+str(i)](temp2)
        
        a['fin'] = self.layers['final'](a['dec1'])
        
        return F.softmax(a['fin'], dim=1)
    
    def describe(self):
        for k,v in self.layers.items():
#            print(k)
            print(k,v)