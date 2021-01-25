import time
from random import randint
import streamlit as st
from PIL import Image
import SessionState
from matching_face_model import MatchingFaceModel
import cv2
import numpy as np
from center.utils.config import Cfg
# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/center")
# print(os.path.dirname(os.path.abspath(__file__)) + "/center")

from detect import CENTER_MODEL

state = SessionState.get(img1_cv = None, img2_cv=None, cmnd_detected=None, init=True, widget_key_1=str(randint(1000, 100000000)), widget_key_2=str(randint(1000, 100000000)),
                         status_cmnd = False)
COSINE_THRESHOLD = 0.7323

def main():
    model_face, model_cmnd = load_model()
    have_cmnd = None
    st.title("ID-Selfie Matching")
    # Load model

    st.sidebar.title("Up ảnh")
    # # img1 = st.sidebar.file_uploader("Img1")
    uploaded_file_1 = st.sidebar.file_uploader("Upload CMND", key=state.widget_key_1)
    uploaded_file_2 = st.sidebar.file_uploader("Upload Selfie ", key=state.widget_key_2)
    if st.sidebar.button("Reset"):
        # state.widget_key_1 = str(randint(1000, 100000000))
        # state.widget_key_2 = str(randint(1000, 100000000))
        uploaded_file_1  = None
        uploaded_file_2 = None
    if uploaded_file_1 is not None:
        print(uploaded_file_1)
        image1 = Image.open(uploaded_file_1)
        st.sidebar.image(image1, use_column_width=True)
        img1_cv = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        state.img1_cv = img1_cv
    if uploaded_file_2 is not None and state.img1_cv is not None:
        image2 = Image.open(uploaded_file_2)
        st.sidebar.image(image2, use_column_width=True)
        img2_cv = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
        state.img2_cv = img2_cv

    # if st.button("Check"):

    if state.img1_cv is not None and state.img2_cv is not None:
        t1 = time.time()
        state.cmnd_detected = None
        state.cmnd_detected, have_cmnd = model_cmnd.detect_obj(state.img1_cv)


        st.title("Kết quả phát hiện CMND")
        if not have_cmnd:
            st.error("Không phát hiện CMND")
            st.image(state.cmnd_detected,  width=312, channels='BGR')
        else:
            st.image(state.cmnd_detected,  width=312, channels='BGR')
        similar_score, img1_aligned, img2_aligned = model_face.matching(state.cmnd_detected, state.img2_cv)
        end_time = round(time.time() - t1, 2)
        if similar_score is not None:
            st.title("Kết quả")
            col1, col2 = st.beta_columns(2)
            with col1:
                st.image(img1_aligned, channels='BGR', width=224)
                st.image(img2_aligned, channels='BGR', width=224)

            with col2:


                percentage, percentage_threshold  = model_face.convert_to_percentage(cosine_score=similar_score, min_val=0.12, max_val=1.65,
                                                                                     cosine_threshold=COSINE_THRESHOLD)

                if percentage >= percentage_threshold:
                    st.success("MATCH")
                else:
                    st.error("NOT MATCH")
                st.success("Độ tương đồng {:.2f}% (threshold={:.2f})".format(percentage, percentage_threshold))
                st.info("Time: {:.2f}s".format(end_time))
                state.img1_cv, state.img2_cv = None, None
                # state.cmnd_detected = None
        else:
            st.error("Không phát hiện khuôn mặt trong ảnh")


    # state.sync()



@st.cache(allow_output_mutation=True)  # hash_func
def load_model(config_path="./center/config/cmnd.yml"):
    print("Loading model ...")
    model_face = MatchingFaceModel(model_path='weights/model-r100-ii/model,0', gpu=0, use_large_detector=True)
    config = Cfg.load_config_from_file(config_path)
    model_cmnd = CENTER_MODEL(config)
    return model_face, model_cmnd


if __name__ == "__main__":
    main()