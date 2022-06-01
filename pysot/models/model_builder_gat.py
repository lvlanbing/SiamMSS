# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.utils.location_grid import compute_locations
from pysot.utils.crop import crop
from pysot.models.attention.S2MPL import S2Attention



class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.attention = S2Attention(256)

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)


    def template(self, z, roi):
        zf = self.backbone(z, roi)
        self.zf = zf

        return zf

    def track(self, x, pre_img, pre_box, init_img, init_bbox, similarity, idx):

        xf = self.backbone(x)

        features = self.attention(xf, self.zf)


        if pre_img is not None and pre_box is not None:

            pre_img = crop(pre_img, pre_box, 127)
            pre_img = [pre_img]

            pre_img = torch.Tensor(pre_img).permute(0,3,1,2)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pre_img = pre_img.to(device)
            pre_xf = self.backbone(pre_img)
            features2 = self.attention(xf, pre_xf)

            #PTB-TIR
            features += features2 * 0.5


        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()
        target_box = data['target_box'].cuda()
        neg = data['neg'].cuda()


        zf = self.backbone(template, target_box)
        xf = self.backbone(search)



        features = self.attention(xf, zf)


        cls, loc, cen = self.car_head(features)

        locations = compute_locations(cls, cfg.TRACK.STRIDE, cfg.TRACK.OFFSET)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc, neg
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss

        return outputs
