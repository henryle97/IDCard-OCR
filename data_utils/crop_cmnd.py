from detect import CENTER_MODEL
import glob
from center.utils.config import Cfg
import cv2
import os
import tqdm


class CMND_Utils:
    def __init__(self, detect_cmnd_config_path="./center/config/cmnd.yml"):
        detect_cmnd_config = Cfg.load_config_from_file(detect_cmnd_config_path)
        self.detect_cmnd_model = CENTER_MODEL(detect_cmnd_config)

    def crop_cmnd(self, cmnd_dir, result_dir):
        if not os.path.exists(result_dir):
            print("Not exist result dir!!!")
            os.mkdir(result_dir)
        paths = glob.glob(cmnd_dir + "/*")
        print(paths)

        for path in tqdm.tqdm(paths):
            try:
                cmnd_img = cv2.imread(path)
                cmnd_cropped, have_cmnd = self.detect_cmnd_model.detect_obj(cmnd_img)
                if have_cmnd:
                    cmnd_cropped_path = os.path.join(result_dir, os.path.basename(path))
                    cv2.imwrite(cmnd_cropped_path, cmnd_cropped)
            except:
                continue
        print("Done crop cmnd")



if __name__ == "__main__":
    utils = CMND_Utils(detect_cmnd_config_path="../center/config/cmnd.yml")
    cmnd_dir = "../data/cmnd_full/images"
    result_dir = "../data/cmnd_full/cropped_imgs"
    utils.crop_cmnd(cmnd_dir, result_dir)


