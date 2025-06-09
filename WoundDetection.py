import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from ChatAPI import wound_chat_assistant, clear_session

# Constants
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Config
st.set_page_config(page_title="Wound Detection App", page_icon="🩹", layout="wide")

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("best_wound_unet_model.keras")

model = load_trained_model()

# Preprocessing
def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    return image / 255.0

# Prediction
def predict_mask(image):
    image_batch = np.expand_dims(image, axis=0)
    pred = model.predict(image_batch)[0]
    pred_thresh = (pred > 0.5).astype(np.uint8)
    if np.sum(pred_thresh) == 0:
        return None
    return pred_thresh

# Overlay color
def overlay_color_on_mask(image, mask, color=(255, 215, 0)):
    mask_rgb = np.stack([mask.squeeze() * c for c in color], axis=-1).astype(np.uint8)
    return cv2.addWeighted(image, 0.6, mask_rgb, 0.4, 0)


# Extract properties
def extract_wound_properties(original, mask):
    props = {}
    binary = mask.squeeze().astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)

        area = np.sum(binary)
        perimeter = cv2.arcLength(largest_contour, True)
        props["Area (pixels)"] = area
        props["Perimeter"] = round(perimeter, 2)

        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            props["Circularity"] = round(circularity, 3)

        masked_pixels = original[binary.astype(bool)]
        if masked_pixels.size > 0:
            mean_color = np.mean(masked_pixels, axis=0)
            props["Mean Color (R,G,B)"] = tuple(map(int, mean_color))

        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        props["Texture Contrast"] = round(contrast, 3)
        props["Texture Homogeneity"] = round(homogeneity, 3)
        # props["Texture Dissimilarity"] = round(dissimilarity, 3)

        edges = cv2.Canny(gray, 100, 200)
        edge_strength = np.mean(edges[binary.astype(bool)])
        props["Edge Sharpness"] = round(edge_strength, 2)

        severity = (contrast + edge_strength + (1 - circularity)) / 3
        props["Wound Severity Index"] = round(severity, 3)

    return props, original


# Streamlit APP UI
st.title("🩹 Wound Detection and Analysis App")

uploaded_file = st.file_uploader("Upload a wound image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_array = preprocess_image(img_pil)

    mask_pred = predict_mask(img_array)
    if mask_pred is None:
        st.warning("⚠️ Sorry, I am unable to detect a wound in the uploaded file.")
    else:
        img_uint8 = (img_array * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)

        img_colored = overlay_color_on_mask(img_rgb.copy(), mask_pred)
        props, image_with_box = extract_wound_properties(img_colored, mask_pred)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Uploaded Wound image**")
            st.image(img_rgb, use_column_width=True)

        with col2:
            st.markdown("**Detected Wound**")
            st.image(image_with_box, use_column_width=True)

        with st.expander("🩻 Wound Properties", expanded=False):
            for k, v in props.items():
                st.markdown(f"- **{k}**: {v}")

        st.markdown("---")

        # LLM Chat
        # clear_session()
        if props is not None:
            wound_chat_assistant(props)
            # wound_chat(props)
