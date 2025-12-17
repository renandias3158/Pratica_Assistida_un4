import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="CNN MNIST", layout="centered")

model = load_model("model/final_CNN_model.h5")

st.title("Reconhecimento de Dígitos Manuscritos (CNN)")
st.write("Envie uma imagem 28x28 em escala de cinza contendo um dígito manuscrito.")

uploaded_file = st.file_uploader("Upload da imagem", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.image(image, caption="Imagem enviada", width=150)
    st.subheader(f"Dígito previsto: {predicted_class}")

    fig, ax = plt.subplots()
    ax.bar(range(10), prediction[0])
    ax.set_xticks(range(10))
    st.pyplot(fig)
