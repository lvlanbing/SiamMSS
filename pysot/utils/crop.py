from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple

import numpy as np
import cv2
# import numbers
# import torch
# import torch.nn as nn


def crop(img, box, out_size):
    # convert box to 0-indexed and center based [y, x, h, w]
    box = np.array([
        box[1] - 1 + (box[3] - 1) / 2,
        box[0] - 1 + (box[2] - 1) / 2,
        box[3], box[2]], dtype=np.float32)
    center, target_sz = box[:2], box[2:]

    # context = 0.5 * np.sum(target_sz)
    context = 0.2 * np.sum(target_sz)
    size = np.sqrt(np.prod(target_sz + context))
    size *= out_size / 127

    avg_color = np.mean(img, axis=(0, 1), dtype=float)
    interp = np.random.choice([
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_NEAREST,
        cv2.INTER_LANCZOS4])
    # patch = crop_and_resize(
    #     img, center, size, out_size,
    #     border_value=avg_color, interp=interp)
    patch = crop_and_resize(
        img, center, size, out_size,
        interp=interp)

    return patch

def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)

    return patch
if __name__ == "__main__":
    img = cv2.imread(r'F:\data\GOT\train_data\GOT-10k_Train_split_01\GOT-10k_Train_000001\00000001.jpg')
    img2 = crop(img, [347.0000, 443.0000, 429.0000, 272.0000], 127)
    cv2.imshow('im', img2)
    cv2.waitKey(600000)