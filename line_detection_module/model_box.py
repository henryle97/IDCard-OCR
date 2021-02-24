from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import uuid
import os
import torch
from line_detection_module.models.networks.pose_dla_dcn import get_pose_net, load_model, pre_process, ctdet_decode, post_process, \
    merge_outputs
import cv2
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

class LineDetection(object):
    def __init__(self, model_weight=''):
        self.num_layers = 34

        self.head_conv = 256
        self.scale = 1.0
        self.threshold = 0.3
        self.num_classes = 5
        self.max_obj_per_img = 100
        self.heads = {'hm': self.num_classes, 'wh': 2, 'reg': 2}
        self.threshold_x = 10
        self.threshold_y = 10
        self.list_label = ['id', 'name', 'date', 'add1', 'add2']
        self.colors = [(255, 0, 0), (0, 0, 255), (123, 2, 190),
                          (253, 124, 98) ,  (255, 251, 134)]

        self.model = get_pose_net(num_layers=self.num_layers, heads=self.heads, head_conv=self.head_conv)

        self.model = load_model(self.model, model_weight)
        if torch.cuda.is_available():
            self.mode = self.model.cuda()


        self.model.eval()

    def predict_box(self, img, show_res=False, return_line_draw=True):
        '''

        @param img: cv2 image: BGR
        @return: list PIL image
        '''

        # Preprocessing image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #convert to rgb (pretrained model RGB)
        image, meta = pre_process(img, self.scale)
        if torch.cuda.is_available():
            image = image.cuda()

        # Predict box
        with torch.no_grad():
            start = time.time()
            output = self.model(image)[-1]
            # print(time.time() - start)
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']
            dets = ctdet_decode(hm, wh, reg=reg, K=100)

        dets = post_process(dets, meta, self.num_classes)
        #print(dets)
        detections = [dets]
        results = merge_outputs(detections, self.num_classes, self.max_obj_per_img)
        # Get boxes with score larger threshold
        h_extend_size = 0.03
        w_extend_size = 0.02
        list_box = {}
        for j in range(1, self.num_classes + 1):
            if self.list_label[j - 1] not in list_box.keys():
                list_box[self.list_label[j - 1]] = []
            for bbox in results[j]:
                if bbox[4] >= self.threshold:
                    print(bbox[4])
                    xmin, ymin, xmax, ymax = max(int(bbox[0]), 0), max(0, int(bbox[1])), \
                                             min(int(bbox[2]),img.shape[1]), min(int(bbox[3]), img.shape[0])

                    # Extend
                    xmin_extend, ymin_extend, xmax_extend, ymax_extend = max(0, xmin - int((xmax - xmin) * w_extend_size)), \
                                             max(0, ymin - int((ymax - ymin) * h_extend_size)), \
                                             xmax + int((xmax - xmin) * w_extend_size), \
                                             ymax + int((ymax - ymin) * h_extend_size)

                    list_box[self.list_label[j-1]].append([xmin_extend, ymin_extend, xmax_extend, ymax_extend])
        print("List box: ", list_box)

        # Show result detect line
        if return_line_draw:
            img_res = Image.fromarray(img)
            img_res = np.ascontiguousarray(img_res)
            # box - [xmin, ymin, xmax, ymax]
            for idx, label in enumerate(self.list_label):

                for box in list_box[label]:
                    cv2.rectangle(img_res, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), self.colors[idx], 2, 2)
            if show_res:
                plt.imshow(img_res)
                plt.show()



        # Crop line
        result_line_img = {}
        img_for_crop = Image.fromarray(img)
        for idx, label in enumerate(self.list_label):
            list_box[label] = sorted(list_box[label], key=lambda box: box[1])
            if label not in result_line_img.keys():
                result_line_img[label] = []
            for box in list_box[label]:
                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]

                line_cropped = img_for_crop.copy().crop((xmin, ymin, xmax, ymax))
                result_line_img[label].append(line_cropped)
        return result_line_img, img_res


