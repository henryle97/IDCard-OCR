import time
import cv2
import streamlit as st
import SessionState
import numpy as np
from PIL import Image
from cmnd_ocr import TEXT_IMAGES
state = SessionState.get(result_text="", res="", prob_positive=0.0, prob_negative= 0.0, initial=True, img_drawed=None, img_cropped=None, reg_text_time=None)


def main():
    model = load_model()
    st.title("Demo nhận dạng văn bản tiếng Việt")
    # Load model

    pages = {
        'CMND': page_cmnd

    }

    st.sidebar.title("Application")
    page = st.sidebar.radio("Chọn ứng dụng demo:", tuple(pages.keys()))

    pages[page](state, model)

    # state.sync()


@st.cache(allow_output_mutation=True)  # hash_func
def load_model():
    print("Loading model ...")
    model = TEXT_IMAGES(reg_model='vgg_seq2seq', ocr_weight_path='weights/seq2seqocr_best.pth')
    return model


def page_cmnd(state, model):
    st.header("Nhận dạng văn bản từ CMND")

    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        pil_image = Image.open(img_file_buffer)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        # print(cv_image.shape)

        # CMND detection
        t1 = time.time()

        result_text, img_drawed_box= model.get_content_image(cv_image)

        state.result_text = result_text
        state.img_drawed = img_drawed_box
        state.reg_text_time = time.time() - t1

        col1, col2 = st.beta_columns(2)
        with col2:

            if state.result_text is not None:
                # result_text_format = []
                # for texts in state.result_text:
                #     result_text_format.append(" ".join(texts))
                st.json(state.result_text)
                st.success("Time: %.2f"%(state.reg_text_time))
            else:
                st.error("Not detected CMND")
        with col1:
            if state.img_drawed is not None:
                st.image(state.img_drawed, use_column_width=True)

        # if state.img_cropped is not None:
        #     st.title("Chi tiết:")
        #     for idx, img in enumerate(state.img_cropped):
        #         st.image(img, caption=state.result_text[idx])
        #         st.empty()

if __name__ == "__main__":
    main()