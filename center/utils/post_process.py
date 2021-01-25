from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .heatmap import transform_preds


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)


def point_post_process(dets, c, s, h, w, num_classes):
  """
    Recover original size of image and predictions
  :param dets:
  :param c:
  :param s:
  :param h:
  :param w:
  :param num_classes:
  :return:
  """
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
      dets[i, :, 0:2], c[i], s[i], (w, h))  # x_center, y_center

    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :2].astype(np.float32),  # score
        dets[i, inds, 2:3].astype(np.float32)], axis=1).tolist()  # class
    ret.append(top_preds)
  return ret

