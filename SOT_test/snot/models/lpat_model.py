# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch as t
import torch.nn as nn
from snot.core.config_lpat import cfg
from snot.models.backbone.alexnet import AlexNet_hift
from snot.models.lpat.utile import LOGO

class ModelBuilderLPAT(nn.Module):
    def __init__(self):
        super(ModelBuilderLPAT, self).__init__()

        self.backbone = AlexNet_hift().cuda()
        self.grader=LOGO(cfg).cuda()
        
    def template(self, z):
        with t.no_grad():
            zf = self.backbone(z)
            self.zf=zf
    
    def track(self, x):
        with t.no_grad():
            xf = self.backbone(x)  
            loc,cls2,cls3=self.grader(xf,self.zf)

            return {
                'cls2': cls2,
                'cls3': cls3,
                'loc': loc
               }