import cv2

from vietocr.tool.translate import build_model, translate, translate_beam_search, process_input, predict, process_batch_input
from vietocr.tool.utils import download_weights
import numpy as np
import matplotlib.pyplot as plt
import torch
from  PIL import Image

class Predictor():
    def __init__(self, config, quanti=False):

        device = config['device']
        
        model, vocab = build_model(config)
        weights = '/tmp/weights.pth'

        if config['weights'].startswith('http'):
            weights = download_weights(config['weights'])
        else:
            weights = config['weights']

        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab

    def quantization_model(self):
        model_quanti = torch.quantization.quantize_dynamic(self.model, dtype=torch.qint8)
        sm = torch.jit.script(model_quanti)
        torch.jit.save(sm, "vgg_seq2seq_quanti.pt")
        self.model = model_quanti
        

    def predict(self, img):
        """
        Predict with one image
        :param img:
        :return:
        """
        img = process_input(img, self.config['dataset']['image_height'], 
                self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])        
        img = img.to(self.config['device'])

        if self.config['predictor']['beamsearch']:
            sent = translate_beam_search(img, self.model)
            s = sent
        else:
            sents = translate(img, self.model)
            s = translate(img, self.model)[0].tolist()

        s = self.vocab.decode(s)

        return s

    def predict_batch(self, batch_img, standard_size):
        batch_img = process_batch_input(batch_img, self.config['dataset']['image_height'], standard_size)
        batch_img = batch_img.to(self.config['device'])
        s = translate(batch_img, self.model).tolist()

        s_decoded = [self.vocab.decode(_) for _ in s]
        return s_decoded


    def predict_with_boxes(self, img, boxes):
        """
        Predict with one image and boxes
        :param img: PIL image
        :param boxes: list: [box1, box2, ..]; box1 = [[{x:..., y:...}] x4]
        :return:
        """
        w, h = img.size

        batch_img = []
        for box in boxes:

            # rescale:
            for g in box:
                g[0]['x'] = g[0]['x'] * w
                g[0]['y'] = g[0]['y'] * h

            [[l1, t1], [r1, t2], [r2, b1], [l2, b2]] = [[box[0][0]['x'], box[0][0]['y']], [box[1][0]['x'], box[1][0]['y']],
                                                        [box[2][0]['x'], box[2][0]['y']], [box[3][0]['x'], box[3][0]['y']]]

            box = [[l1, t1], [r1, t2], [r2, b1], [l2, b2]]

            if l1 != l2 or t1 != t2 or r1 != r2 or b1 != b2:
                # polyline : transform
                crop_transform_img = self.transform_image(np.array(img), np.array(box, dtype='float32'))
                # plt.imshow(crop_transform_img)
                # plt.show()
                batch_img.append(Image.fromarray(crop_transform_img))
                # batch_img[-1].save('check.jpg')
            else:
                xmin = l1
                ymin = t1
                xmax = r1
                ymax = b1
                crop_img =img.copy().crop((max(0, xmin), max(0, ymin),
                                     xmax, ymax))

                batch_img.append(crop_img)

        res_pred = self.predict_batch(batch_img)

        return res_pred


    def transform_image(self, image, rect):
        """

        :param image: cv2
        :param rect: numpy array
        :return: cv2
        """
        # print(type(rect))
        # print(rect)
        [tl, tr, br, bl] = rect
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

