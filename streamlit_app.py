import numpy as np
import streamlit as st
from PIL import Image

from car_detector import predict_image


def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

# set title
TITLE = """
<h1 style='text-align: center;'>CAR DETECTOR USING R-CNN</h1>
"""
THRESHOLD_LIST = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])

def create_ui():
    # title
    st.markdown(TITLE, unsafe_allow_html=True)
    # st.write("Hello :)))")
    image_file = st.file_uploader("Choose Image to detect", type=["png", "jpg", "jpeg"])
    threshold = st.selectbox("Select the threshold", options=THRESHOLD_LIST,index=2)
    detect_button = st.button("Detect")
    col1, col2, col3 = st.columns([1, 6, 1])
    st.write("Threshold = " + str(threshold))
    with col1:
        pass
    with col2:
        if image_file is not None:
            # To View Uploaded Image
            img = load_image(image_file)
            col1.image(img, width=250,use_column_width=True)
    with col3:
        pass
    if detect_button:
        if image_file is not None:
            st.write("Waiting for detecting ...")
            img_predicted = predict_image(img, threshold)
            st.image(img_predicted, width=250, use_column_width=True)
        else:
            st.write("You have to choose image to detect.")


if __name__ == "__main__":
    create_ui()