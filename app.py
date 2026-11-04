import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
@st.cache_resource
def get_model():
    model = load_model('keras_model.h5')
    return model

model = get_model()

st.title("🌱 AI Smart Waste Segregation & Recycling Assistant")
st.write("Upload any waste or item image to identify its category and recycling method!")

file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    
    # 6 Classes mapping
    class_names = ['Plastic', 'Paper & Cardboard', 'Organic Waste', 'Glass', 'Metal & Scrap', 'E-Waste & Others']
    predicted_class = class_names[index]
    confidence_score = round(prediction[0][index] * 100, 2)

    st.success(f"Result: This belongs to **{predicted_class}** category! (Confidence: {confidence_score}%)")
    
    # Dynamic Recycling Tips
    if predicted_class == 'Plastic':
        st.info("💡 Tip: Clean the plastic and send it to local plastic recycling units.")
    elif predicted_class == 'Paper & Cardboard':
        st.info("💡 Tip: Keep it dry and send it for paper recycling/scrap collection.")
    elif predicted_class == 'Organic Waste':
        st.info("💡 Tip: Perfect for composting or converting into organic manure.")
    elif predicted_class == 'Glass':
        st.info("💡 Tip: Handle with care and drop it in glass recycling bins.")
    elif predicted_class == 'Metal & Scrap':
        st.info("💡 Tip: Collect and sell to local metal scrap vendors for recycling.")
    else:
        st.info("💡 Tip: E-Waste and other items require special electronic recycling centers.")