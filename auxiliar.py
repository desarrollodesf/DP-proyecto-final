from pathlib import Path
import cv2
import numpy as np
from tensorflow import keras
import streamlit as st
import pandas as pd
from roboflow import Roboflow

img_height = 224
img_width = 224

def decode_image(img, name):
    image = cv2.imdecode(
        np.frombuffer(img, dtype=np.uint8), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    path = f"img/{name}"
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return path

@st.experimental_singleton()
def load_model():
    return keras.models.load_model('Xception')

def load_model2():
    rf = Roboflow(api_key="sB8KCdZOn2R0K8TOOT0n")
    project = rf.workspace().project("monkeypox-project-cnn")
    model = project.version(3).model
    return model

def predict2(image_path):
    image = st.session_state['detection'].predict(image_path, confidence=40, overlap=30)
    image.save(output_path=image_path)
    return image_path


def predict(image_path):
    image = keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    y_pred = st.session_state['model'].predict(input_arr)
    y_pred = pd.DataFrame(y_pred, columns=['Monkeypox', 'Others'])
    y_pred = y_pred * 100
    return image, y_pred