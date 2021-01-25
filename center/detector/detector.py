from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.heatmap import get_affine_transform
from utils.post_process import point_post_process

class BaseDetector(object):
    def __init__(self, config):

        self.mean = np.array(config['dataset']['mean'], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(config['dataset']['std'], dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = config['dataset']['max_object']
        self.num_classes = config['dataset']['num_classes']
        self.input_h = config['model']['input_h']
        self.input_w = config['model']['input_w']
        self.pad = config['model']['pad']
        self.down_ratio = config['model']['down_ratio']
        self.fix_res = config['predictor']['fix_res']

        print("BaseDetector CONFIG: ", config)

    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]        # image.shape = height, width, channel
        new_height = int(height * scale)
        new_width = int(width * scale)

        if self.fix_res:        # default True
            inp_height, inp_width = self.input_h, self.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)       # center
            s = max(height, width) * 1.0                                            # scale
        else:
            inp_height = (new_height | self.pad) + 1
            inp_width = (new_width | self.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        # Get transform matrix
        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])  # input: center, scale, rotate,  output_size

        resized_image = cv2.resize(image, (new_width, new_height))

        inp_image = cv2.warpAffine(                     # return shape 512x512x3
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)

        # normalize input
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)      # chanel, height, width

        images = torch.from_numpy(images)

        meta = {'c': c, 's': s,
                'out_height': inp_height // self.down_ratio,
                'out_width': inp_width // self.down_ratio}
        return images, meta

    def post_process(self, dets, meta, scale=1):
        """
        Phục hồi prediction về kích thước ảnh đầu vào ban đầu
        :param dets:
        :param meta:
        :param scale:
        :return:
        """
        # from IPython import embed; embed()
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])  # [1, 2000, 4]
        dets = point_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 3)  # [0-3: coord, 4:score, 5:cls]
            dets[0][j][:, :2] /= scale  # x_c, y_c
        return dets[0]

    def merge_outputs(self, detections):
        # from IPython import embed;
        # embed()
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            # if len(self.scales) > 1 or self.opt.nms:
            #    soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 2] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 2] >= thresh)
                results[j] = results[j][keep_inds]
        return results
