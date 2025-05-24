import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model (legacy HDF5 format)
try:
    model = load_model("retina_oct_model.h5")
    model_loaded = True
except Exception as e:
    st.error("❌ Error loading model. Check if 'retina_oct_model.h5' exists.")
    st.stop()

# Label map for class prediction
label_map = {0: 'Normal', 1: 'CNV', 2: 'DME', 3: 'DRUSEN'}

# UI Setup
st.title("🔬 Retina Damage Detection via OCT Images")
st.markdown("Upload an **OCT scan image** to analyze retina health and predict possible eye conditions.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload an OCT image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    img = Image.open(uploaded_file).resize((224, 224)).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Prediction
    prediction = model.predict(img_arr)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100

    # Display result
    st.success(f"🧠 Prediction: **{label_map[predicted_class]}**")
    st.info(f"🔎 Confidence: **{confidence:.2f}%**")

    # Show class probabilities
    st.subheader("🔢 Prediction Breakdown")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{label_map[i]}: {prob * 100:.2f}%")

# Footer
st.caption("Model trained on dummy data. Replace with real OCT data for medical use.")
