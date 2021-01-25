import os
import torch

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "center"))
# print(os.path.join(os.path.dirname(os.path.abspath(__file__)), "center"))
# print(sys.path)

from models.model import load_model, create_model
from detector.detector import BaseDetector
from models.decode import point_decode
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from utils.config import Cfg
from utils.debugger import Debugger
import argparse


class CENTER_MODEL(object):
    def __init__(self, config):

        self.weight_path = config['predictor']['weight']
        self.scale = config['predictor']['scale']
        self.threshold = config['predictor']['threshold']
        self.max_obj_predict = config['dataset']['max_object']
        self.num_classes = config['dataset']['num_classes']
        self.arch = config['model']['arch']
        self.heads = config['model']['heads']
        self.head_conv = config['model']['head_conv']

        self.model = create_model(arch=self.arch, heads=self.heads, head_conv=self.head_conv)
        self.model = load_model(self.model, self.weight_path)

        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        self.dt = BaseDetector(config)
        self.debugger = Debugger(num_classes=self.num_classes)

    def detect_obj(self, img, show=False, save_res=False):
        """
        Fix resolution image
        :param img: cv2 image
        :return: cv2 image
        """
        image, meta = self.dt.pre_process(img, self.scale)  # size image change to 512x512x3
        # from IPython import embed; embed();
        with torch.no_grad():
            if torch.cuda.is_available():
                image = image.cuda()
            else:
                image = image.to(torch.device('cpu'))
            start = time.time()
            output = self.model(image)[-1]
            # print("Time predict: ", time.time() - start)
            hm = output['hm'].sigmoid_()
            reg = output['reg']
            dets = point_decode(hm, reg=reg, K=self.max_obj_predict)

        dets = self.dt.post_process(dets, meta)
        dets = [dets]
        results = self.dt.merge_outputs(dets)

        list_center = []
        qh_box = []
        for j in range(1, self.num_classes + 1):

            if j == 2:  # quochuy
                for bbox in results[j]:
                    if bbox[2] >= self.threshold:
                        x_center, y_center = max(int(bbox[0]), 0), max(0, int(bbox[1]))
                        qh_box.append([x_center, y_center])
                        break
            else:

                for bbox in results[j]:
                    if bbox[2] >= self.threshold:
                        x_center, y_center = max(int(bbox[0]), 0), max(0, int(bbox[1]))
                        list_center.append([x_center, y_center])

        if show:
            img_draw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            colors = {'red': (255, 0, 0), 'blue': (0, 0, 255)}
            for center in list_center:
                img_draw = cv2.circle(img_draw, (center[0], center[1]), radius=2, color=colors['red'], thickness=2)
            if len(qh_box) > 0:
                img_draw = cv2.circle(img_draw, (qh_box[0][0], qh_box[0][1]), radius=2, color=colors['blue'],
                                      thickness=2)
            # cv2.imwrite('img_draw', img_draw)
            plt.imshow(img_draw)
            plt.show()

        if (len(list_center) == 4):
            points = self.order_points(np.array(list_center[:4]))
        else:
            # print("Cannot detect 4 corners !!!, Number of conners detected was ", len(list_center))
            return img, False

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_aligh = self.align(img, points, qh_box)

        if show:
            plt.imshow(cv2.cvtColor(img_aligh, cv2.COLOR_BGR2RGB))
            plt.show()

        if save_res:
            cv2.imwrite("../insightface/deploy/imgs/res.jpg", img_aligh)

        return img_aligh, True

    def order_points(self, pts):

        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        if rect[1][1] > rect[3][1]:
            tmp = rect[1]
            rect[1] = rect[3]
            rect[3] = tmp
        return rect

    def post_sort_order(self, points, qh_corr):
        c1, c2, c3, c4 = points[0], points[1], points[2], points[3]
        # print(c1, c2, c3, c4, qh_corr, sep="\n")
        if self.dist(c1, c2) < self.dist(c1, c4):  # vertical image
            if self.dist(qh_corr, c2) < self.dist(qh_corr, c4):
                return np.array([c2, c3, c4, c1])
            else:
                return np.array([c4, c1, c2, c3])
        else:
            if self.dist(qh_corr, c1) < self.dist(qh_corr, c3):
                return points
            else:
                return np.array([c3, c4, c1, c2])

    def dist(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def align(self, image, pts, qh_coor):
        pts = np.array(pts, dtype="float32")
        rect = self.order_points(pts)
        rect = self.post_sort_order(rect, qh_coor)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./center/config/cmnd.yml')
    parser.add_argument("--image_path", type=str,
                        default='C:\\Users\\hoanglv10\\PycharmProjects\\Object_Corner_Detection\\demo\\cmnd_hoang.jpg')
    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)
    # print(config)
    model = CENTER_MODEL(config)
    # for i in range(10):
    img = cv2.imread(args.image_path)
    model.detect_obj(img, show=False, save_res=False)

"""
[ 156. 1524.]
[1554. 1037.]
[1863. 1930.]
[ 444. 2413.]
[[443, 1617]]
"""
