import os
import sys
import numpy as np
import cv2 as cv
import streamlit as st
from streamlit_webrtc import VideoProcessorBase

cd = os.getcwd()
model = os.path.join(cd, "models/res10_300x300_ssd_iter_140000.caffemodel")
proto = os.path.join(cd, "models/configs/deploy.prototxt")

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]


model = cv.dnn.readNetFromCaffe(prototxt=proto,
                                caffeModel=model)

dim = 300
mean = [104, 177, 123]
conf_thresh = 0.2


class VideoTransformer(VideoProcessorBase):
    def transform(self, img):
        stframe = st.empty()
        frames = []    
        frame = img.to_ndarray(format="bgr24")    
        frame = cv.flip(frame, 1)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        
        blob = cv.dnn.blobFromImage(frame, 0.5, (dim, dim), mean, swapRB=False, crop=False)
        
        model.setInput(blob=blob)
        detections = model.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_thresh:
                x_bottom_left = int(detections[0, 0, i, 3] * frame_width)
                y_bottom_left = int(detections[0, 0, i, 4] * frame_height)
                x_top_right   = int(detections[0, 0, i, 5] * frame_width)
                y_top_right   = int(detections[0, 0, i, 6] * frame_height)
                
                cv.rectangle(frame, (x_bottom_left, y_bottom_left), (x_top_right, y_top_right), (255, 117, 234), 2)
                
                label = "Confidence: {:.2f}%".format(confidence * 100)
                label_size, baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, .5, 2)
                cv.rectangle(frame, (x_bottom_left, y_bottom_left - label_size[1]),
                            (x_bottom_left + label_size[0], y_bottom_left + baseline), (255, 255, 243), cv.FILLED)
                cv.putText(frame, label, (x_bottom_left, y_bottom_left), cv.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 0, 0), 1)
                
        t, _ = model.getPerfProfile()
        label = "Inference time: %.2f ms" % (t * 1000 / cv.getTickFrequency())
        cv.putText(frame, label, (5, 15), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255))
        frames.append(frame)
        stframe.image(frame, "Say Cheese", channels="BGR")
        
        return frame

def detect_faces(img):
    image = np.array(img)
    h, w = image.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(image, (dim, dim)), 1.0, (dim, dim), mean, swapRB=False, crop=False)
    
    model.setInput(blob)
    detections = model.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, endX, startY, endY) = box.astype("int")
            cv.rectangle(image, (startX - 10, endX), (startY, endY + 20), (0, 255, 0), 2)
            
            label = "Confidence: {:.2f}%".format( confidence * 100)
            label_size, baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, .5, 2)
            cv.rectangle(image, (startX - 10, endX - label_size[1]),
                        (startX + label_size[0], endX + baseline), (255, 255, 243), cv.FILLED)
            cv.putText(image, label, (startX, endX), cv.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 0), 1)
    return image

def real_timeDetection(model=model, source=s):
    cap = cv.VideoCapture(source)
    stframe = st.empty()
    frames = []
    frame_dict = {}
    
    if not cap.isOpened():
        st.error("There a problem while trying to access your webcam.")   
         
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv.flip(frame, 1)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        
        blob = cv.dnn.blobFromImage(frame, 0.5, (dim, dim), mean, swapRB=False, crop=False)
        # blobGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        model.setInput(blob=blob)
        detections = model.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_thresh:
                x_bottom_left = int(detections[0, 0, i, 3] * frame_width)
                y_bottom_left = int(detections[0, 0, i, 4] * frame_height)
                x_top_right   = int(detections[0, 0, i, 5] * frame_width)
                y_top_right   = int(detections[0, 0, i, 6] * frame_height)
                
                cv.rectangle(frame, (x_bottom_left, y_bottom_left), (x_top_right, y_top_right), (255, 117, 234), 2)
                
                label = "Confidence: {:.2f}%".format(confidence * 100)
                label_size, baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, .5, 2)
                cv.rectangle(frame, (x_bottom_left, y_bottom_left - label_size[1]),
                            (x_bottom_left + label_size[0], y_bottom_left + baseline), (255, 255, 243), cv.FILLED)
                cv.putText(frame, label, (x_bottom_left, y_bottom_left), cv.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 0, 0), 1)
                
        t, _ = model.getPerfProfile()
        label = "Inference time: %.2f ms" % (t * 1000 / cv.getTickFrequency())
        cv.putText(frame, label, (5, 15), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255))
        frames.append(frame)
        stframe.image(frame, "Say Cheese", channels="BGR")
    
    cap.release()
    
    return frames
