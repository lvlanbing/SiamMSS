"""
This file contains specific functions for computing losses of SiamCAR
file
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import math
from pysot.models.metrics import bbox_iou

INF = 100000000


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class DIOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)

        # cal outer boxes
        outer_w = torch.max(pred_left, target_left) + \
                      torch.max(pred_right, target_right)
        outer_h = torch.max(pred_bottom, target_bottom) + \
                      torch.max(pred_top, target_top)
        outer_diagonal_line = outer_w.pow(2) + outer_h.pow(2)
        # outer_w = np.maximum(outer_w, 0.0)
        # outer_h = np.maximum(outer_h, 0.0)
        # outer_diagonal_line = np.square(outer_w) + np.square(outer_h)

        # cal center distance
        boxes1_cx = (target_left + target_right) * 0.5
        boxes1_cy = (target_top + target_bottom) * 0.5
        boxes2_cx = (pred_left + pred_right) * 0.5
        boxes2_cy = (pred_top + pred_bottom) * 0.5
        center_dis = (boxes1_cx - boxes2_cx).pow(2) + (boxes1_cy - boxes2_cy).pow(2)

        # cal diou
        dious = ious - center_dis / outer_diagonal_line

        losses = 1 - dious
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class GIOULoss(nn.Module):
    def __init__(self, loc_loss_type='giou'):
        super(GIOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion

        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

# 下午工作内容 修改CIOU，进一步修改generalized focal loss V1, V2
class CIOULoss(nn.Module):
    def __init__(self, loc_loss_type='ciou'):
        super(CIOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type
    def forward(self, pred, target, weight=None):
        # 这里给的中心点和高宽，需要修改为左上右下两个坐标
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]
        pred_h = (pred_right + pred_left)
        pred_w = (pred_bottom + pred_top)

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]
        target_h = (target_right + target_left)
        target_w = (target_bottom + target_top)

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)

        # cal outer boxes
        outer_w = torch.max(pred_left, target_left) + \
                  torch.max(pred_right, target_right)
        outer_h = torch.max(pred_bottom, target_bottom) + \
                  torch.max(pred_top, target_top)
        outer_diagonal_line = outer_w.pow(2) + outer_h.pow(2)
        # outer_w = np.maximum(outer_w, 0.0)
        # outer_h = np.maximum(outer_h, 0.0)
        # outer_diagonal_line = np.square(outer_w) + np.square(outer_h)

        # cal center distance
        boxes1_cx = (target_left + target_right) * 0.5
        boxes1_cy = (target_top + target_bottom) * 0.5
        boxes2_cx = (pred_left + pred_right) * 0.5
        boxes2_cy = (pred_top + pred_bottom) * 0.5
        center_dis = (boxes1_cx - boxes2_cx).pow(2) + (boxes1_cy - boxes2_cy).pow(2)

        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(target_w/target_h) - torch.atan(pred_w/pred_h)), 2)
        with torch.no_grad():
            S = (ious > 0.5).float()
            alpha = (S * v) / (1-ious+v)

        # cal ciou
        cious = ious - center_dis / outer_diagonal_line - (alpha * v)

        losses = 1 - cious
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

# def ciou(bboxes1, bboxes2):
#     bboxes1 = torch.sigmoid(bboxes1)
#     bboxes2 = torch.sigmoid(bboxes2)
#     rows = bboxes1.shape[0]
#     cols = bboxes2.shape[0]
#     cious = torch.zeros((rows, cols))
#     if rows * cols == 0:
#         return cious
#     exchange = False
#     if bboxes1.shape[0] > bboxes2.shape[0]:
#         bboxes1, bboxes2 = bboxes2, bboxes1
#         cious = torch.zeros((cols, rows))
#         exchange = True
#     w1 = torch.exp(bboxes1[:, 2])
#     h1 = torch.exp(bboxes1[:, 3])
#     w2 = torch.exp(bboxes2[:, 2])
#     h2 = torch.exp(bboxes2[:, 3])
#     area1 = w1 * h1
#     area2 = w2 * h2
#     center_x1 = bboxes1[:, 0]
#     center_y1 = bboxes1[:, 1]
#     center_x2 = bboxes2[:, 0]
#     center_y2 = bboxes2[:, 1]
#
#     inter_l = torch.max(center_x1 - w1 / 2,center_x2 - w2 / 2)
#     inter_r = torch.min(center_x1 + w1 / 2,center_x2 + w2 / 2)
#     inter_t = torch.max(center_y1 - h1 / 2,center_y2 - h2 / 2)
#     inter_b = torch.min(center_y1 + h1 / 2,center_y2 + h2 / 2)
#     inter_area = torch.clamp((inter_r - inter_l),min=0) * torch.clamp((inter_b - inter_t),min=0)
#
#     c_l = torch.min(center_x1 - w1 / 2,center_x2 - w2 / 2)
#     c_r = torch.max(center_x1 + w1 / 2,center_x2 + w2 / 2)
#     c_t = torch.min(center_y1 - h1 / 2,center_y2 - h2 / 2)
#     c_b = torch.max(center_y1 + h1 / 2,center_y2 + h2 / 2)
#
#     inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
#     c_diag = torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2
#
#     union = area1+area2-inter_area
#     u = (inter_diag) / c_diag
#     iou = inter_area / union
#     v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
#     with torch.no_grad():
#         S = (iou > 0.5).float()
#         alpha= S*v/(1-iou+v)
#     cious = iou - u - alpha * v
#     cious = torch.clamp(cious,min=-1.0,max = 1.0)
#     if exchange:
#         cious = cious.T
#     return torch.sum(1-cious)

class SiamCARLossComputation(object):
    """
    This class computes the SiamCAR losses.
    """

    def __init__(self,cfg):
        # self.box_reg_loss_func = CIOULoss()
        self.box_reg_loss_func = IOULoss()
        # self.box_reg_loss_func = GIOULoss()
        # self.box_reg_loss_func = bbox_iou()
        # self.box_reg_loss_func = DIOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.cfg = cfg

    def prepare_targets(self, points, labels, gt_bbox, neg):

        labels, reg_targets, pos_area = self.compute_targets_for_locations(
            points, labels, gt_bbox, neg
        )

        return labels, reg_targets, pos_area

    def compute_targets_for_locations(self, locations, labels, gt_bbox, neg):
        # reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        bboxes = gt_bbox
        labels = labels.view(self.cfg.TRAIN.OUTPUT_SIZE**2, -1)
        pos_area = torch.zeros_like(labels)

        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

##########################################points in the gt_bbox area################################################
        all_s1 = reg_targets_per_im[:, :, 0] > 0
        all_s2 = reg_targets_per_im[:, :, 2] > 0
        all_s3 = reg_targets_per_im[:, :, 1] > 0
        all_s4 = reg_targets_per_im[:, :, 3] > 0
        all_in_boxes = all_s1 * all_s2 * all_s3 * all_s4
        all_pos = np.where(all_in_boxes.cpu() == 1)
        pos_area[all_pos] = 1

##########################################ignore labels################################################
        ignore_s1 = reg_targets_per_im[:, :, 0] > 0.2 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
        ignore_s2 = reg_targets_per_im[:, :, 2] > 0.2 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
        ignore_s3 = reg_targets_per_im[:, :, 1] > 0.2 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
        ignore_s4 = reg_targets_per_im[:, :, 3] > 0.2 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
        ignore_in_boxes = ignore_s1 * ignore_s2 * ignore_s3 * ignore_s4
        ignore_pos = np.where(ignore_in_boxes.cpu() == 1)
        labels[ignore_pos] = -1

        s1 = reg_targets_per_im[:, :, 0] > 0.5*((bboxes[:,2]-bboxes[:,0])/2).float()
        s2 = reg_targets_per_im[:, :, 2] > 0.5*((bboxes[:,2]-bboxes[:,0])/2).float()
        s3 = reg_targets_per_im[:, :, 1] > 0.5*((bboxes[:,3]-bboxes[:,1])/2).float()
        s4 = reg_targets_per_im[:, :, 3] > 0.5*((bboxes[:,3]-bboxes[:,1])/2).float()
        is_in_boxes = s1*s2*s3*s4
        pos = np.where(is_in_boxes.cpu() == 1)
        labels[pos] = 1
        labels = labels * (1 - neg.long())

        return labels.permute(1, 0).contiguous(), reg_targets_per_im.permute(1, 0, 2).contiguous(), pos_area.permute(1, 0).contiguous()

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])

        left, top, right, bottom = reg_targets[:, 0], reg_targets[:, 1], reg_targets[:, 2], reg_targets[:, 3]
        # 修改中心点的计算 2021/7/21
        size = centerness.shape
        for i in range(size[0]):

            if centerness[i] < 0.2:
                centerness[i] = 0
            # else:
            #     centerness[i] += (right[i] - left[i]) ** 2 + (bottom[i] - top[i]) ** 2
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, labels, reg_targets, neg):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """

        label_cls, reg_targets, pos_area = self.prepare_targets(locations, labels, reg_targets, neg)
        box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4))
        labels_flatten = (label_cls.view(-1))
        reg_targets_flatten = (reg_targets.view(-1, 4))
        centerness_flatten = (centerness.view(-1))

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        ###########################change cen and reg area###################################
        pos_area_flatten = (pos_area.view(-1))
        all_pos_idx = torch.nonzero(pos_area_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[all_pos_idx]
        reg_targets_flatten = reg_targets_flatten[all_pos_idx]
        centerness_flatten = centerness_flatten[all_pos_idx]

        #####################################################################################

        # box_regression_flatten = box_regression_flatten[pos_inds]
        # reg_targets_flatten = reg_targets_flatten[pos_inds]
        # centerness_flatten = centerness_flatten[pos_inds]
        cls_loss = select_cross_entropy_loss(box_cls, labels_flatten)

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            # reg_loss = ciou(box_regression_flatten, reg_targets_flatten)
            # reg_loss = bbox_iou(box_regression_flatten, reg_targets_flatten, x1y1x2y2=True,CIoU=True).mean()
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


def make_siamcar_loss_evaluator(cfg):
    loss_evaluator = SiamCARLossComputation(cfg)
    return loss_evaluator
