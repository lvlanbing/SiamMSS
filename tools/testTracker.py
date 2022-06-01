# Copyright (c) SenseTime. All Rights Reserved.

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import math
import sys
env_path = os.path.join(os.path.dirname(__file__), '..')

import copy
import collections


sys.path.append('../')
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pysot.core.config import cfg
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder_gat import ModelBuilder
from toolkit.datasets import DatasetFactory
from pysot.tracker.siamgat_tracker import SiamGATTracker
from pysot.utils.crop import crop
from toolkit.utils.region import vot_overlap, vot_float2str


parser = argparse.ArgumentParser(description='siamsmm tracking')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--dataset', type=str, default='VOTTIR2015',
        help='datasets') # OTB100 LaSOT UAV123 GOT-10k
parser.add_argument('--vis', action='store_true', default=True,
        help='whether visualzie result')
parser.add_argument('--snapshot', type=str, default=r'F:\program\SiamGAT-main\SiamGAT-main\tools\snapshot/checkpoint_e19_S2_MPL.pth',
        help='snapshot of models to eval')
parser.add_argument('--config', type=str, default=r'F:\program\SiamGAT-main\SiamGAT-main/experiments/siamgat_googlenet_got10k/config.yaml',
        help='config file')
args = parser.parse_args()

torch.set_num_threads(1)



def iou(pred, target):
    pred_x, pred_y, pred_w, pred_h = pred
    target_x, target_y, target_w, target_h = target

    pred_area = pred_w * pred_h
    target_area = target_h * pred_w

    w = min(target_x + target_w, pred_x + pred_w) - max(pred_x, target_x)
    h = min(target_y + target_h, pred_y + pred_h) - max(pred_y, target_y)

    if w <= 0 or h <= 0:
        return 0

    area_c = w * h
    return area_c / (pred_area + target_area - area_c)


def main():
    # load config
    cfg.merge_from_file(args.config)

    # Test dataset
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # set hyper parameters
    params = getattr(cfg.HP_SEARCH, args.dataset)
    cfg.TRACK.LR = params[0]
    cfg.TRACK.PENALTY_K = params[1]
    cfg.TRACK.WINDOW_INFLUENCE = params[2]

    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SiamGATTracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[-2]


    total_lost = 0

    if args.dataset in ["VOTTIR2015", "VOTTIR2017"]:

        for v_idx, video in enumerate(dataset):
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            pre_img = None
            init_img = None
            pre_box = None
            init_bbox = None
            similarity = 0
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()

                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                    pre_img = None

                elif idx > frame_counter:
                    outputs = tracker.track(img,pre_img, pre_box, init_img, init_bbox, similarity, idx)
                    pred_bbox = outputs['bbox']
                    pre_box = outputs['bbox']
                    pre_img = outputs['pre_img']

                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)

                toc += cv2.getTickCount() - tic

                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            video_path = os.path.join('results', args.dataset, model_name,
                                      'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))

    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            # if video.name != "distractor2":
            #     continue
            toc = 0
            pred_bboxes = []
            track_times = []
            pre_img = None
            pre_box = None
            init_img = None
            init_bbox = None
            similarity = 0
            for idx, (img, gt_bbox) in enumerate(video):

                tic = cv2.getTickCount()

                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(pred_bbox)
                    pre_img = None
                    pre_box = None
                    init_img = img
                    init_bbox = gt_bbox_
                else:
                    # 跟踪
                    outputs = tracker.track(img, pre_img, pre_box, init_img, init_bbox, similarity, idx)
                    pred_bbox = outputs['bbox']

                    pre_box = outputs['bbox']
                    pre_img = outputs['pre_img']
                    pred_bboxes.append(pred_bbox)


                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    if not any(map(math.isnan,gt_bbox)):
                        gt_bbox = list(map(int, gt_bbox))
                        pred_bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                      (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                        cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                      (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                        cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.imshow(video.name, img)
                        cv2.waitKey(1)
            toc /= cv2.getTickFrequency()

            # save results
            if 'GOT_10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
