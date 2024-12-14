import streamlit as st
from process import extract_characters
import os

st.title("Handwritten Sentence Recognition")

uploaded_file = st.file_uploader("Upload an image (PNG/JPG/JPEG)", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = "temp_uploaded.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Run inference
    sentence = extract_characters(temp_path)
    st.image(temp_path, caption="Uploaded Image")
    st.write("Recognized sentence:")
    st.write(sentence)

    # Cleanup (optional)
    os.remove(temp_path)
