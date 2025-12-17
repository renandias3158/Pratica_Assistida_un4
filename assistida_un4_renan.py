import os
import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

st.set_page_config(page_title="CNN com TensorFlow Lite", layout="centered")

MODEL_H5 = "model/modelo_cnn.h5"
MODEL_TFLITE = "model/modelo_cnn.tflite"

if not os.path.exists(MODEL_TFLITE):
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_H5)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(MODEL_TFLITE, "wb") as f:
        f.write(tflite_model)

interpreter = tflite.Interpreter(model_path=MODEL_TFLITE)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"][1:3]

st.title("Classificação de Imagem com CNN")

file = st.file_uploader("Envie uma imagem", type=["png", "jpg", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    image = image.resize(input_shape)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    st.image(image, use_container_width=True)
    st.write(f"Classe prevista: {predicted_class}")
    st.write(f"Confiança: {confidence:.2%}")
