import streamlit as st
import numpy as np
from PIL import Image
import cv2 as cv
import tempfile
import os

from cafFace import real_timeDetection, detect_faces

cd = os.getcwd()
default = os.path.join(cd, 'data/videos/faces1.mp4')

st.set_page_config(page_title="Face Detector")

st.title("Real-Time Face Detection: Capture faces in image or video frames.")
st.page_link("https://ririnjaramba.onrender.com", label=":blue-background[Developer Portfolio]", icon=":material/thumb_up:")
st.text("Pro-tip: Try it with your webcam.")

option = st.selectbox("Choose Input Type", ("----- Select One ------", "Camera", "Upload Video", "Upload Image"))

if option == "----- Select One ------":
    real_timeDetection(source=default)
    
    if st.button('Re-run'):
        st.rerun()

elif option == "Camera":
    real_timeDetection()
    
elif option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image = np.array(image)
            image = detect_faces(image)
            st.image(image, caption="Uploaded image", use_column_width=True)
        except Exception as e:
            st.error(f"An error occured: {e}")
                    
elif option == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video file...", type=['mp4', 'webm', 'avi', 'mov'])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tfile:
            tfile.write(uploaded_file.read())
            tfile.close()
        
        frames = real_timeDetection(source=tfile.name)
        
        if st.button('Re-run'):
            st.rerun()
        
        # out_path = tfile.name.replace(os.path.splitext(tfile.name)[1], '_detected.mp4')
        # height, width, _ = frames[0].shape
        # fourcc = cv.VideoWriter_fourcc(*'mp4v')
        # out = cv.VideoWriter(out_path, fourcc, 20.0, (width, height))
        
        # for frame in frames:
        #     out.write(cv.cvtColor(np.array(frame), cv.COLOR_RGB2BGR))
        # out.release()
                
        # st.video(out_path, format="video/mp4", start_time=0)
        
        try:
            os.remove(tfile.name)
        except FileNotFoundError:
            pass
        
        # try:
        #     os.remove(out_path)
        # except FileNotFoundError:
        #     pass
