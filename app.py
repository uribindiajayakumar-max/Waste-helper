import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="AI Smart Waste Helper",
    page_icon="♻️",
    layout="centered"
)

# ---------------------------------
# LOAD LABELS
# ---------------------------------

@st.cache_data
def load_labels():
    with open("labels.txt", "r", encoding="utf-8") as file:
        labels = [line.strip() for line in file if line.strip()]

    # Remove Teachable Machine class numbers
    cleaned_labels = []

    for label in labels:
        parts = label.split(maxsplit=1)

        if len(parts) == 2 and parts[0].isdigit():
            cleaned_labels.append(parts[1])
        else:
            cleaned_labels.append(label)

    return cleaned_labels


# ---------------------------------
# CUSTOM LAYER FOR OLD MODEL
# ---------------------------------

class CompatibleDepthwiseConv2D(
    tf.keras.layers.DepthwiseConv2D
):
    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)


# ---------------------------------
# LOAD TRAINED MODEL
# ---------------------------------

@st.cache_resource
def load_waste_model():

    model = tf.keras.models.load_model(
        "keras_model.h5",
        custom_objects={
            "DepthwiseConv2D": CompatibleDepthwiseConv2D,
            "CompatibleDepthwiseConv2D": CompatibleDepthwiseConv2D
        },
        compile=False
    )

    return model


# ---------------------------------
# PREDICTION FUNCTION
# ---------------------------------

def predict_waste(image, model, labels):

    image = image.convert("RGB")

    input_shape = model.input_shape

    height = input_shape[1] or 224
    width = input_shape[2] or 224

    image = ImageOps.fit(
        image,
        (width, height),
        method=Image.Resampling.LANCZOS
    )

    image_array = np.asarray(image).astype(np.float32)

    # Teachable Machine normalization
    normalized_image = (image_array / 127.5) - 1

    input_data = np.expand_dims(normalized_image, axis=0)

    predictions = model.predict(input_data, verbose=0)

    predictions = np.asarray(predictions).flatten()

    predicted_index = int(np.argmax(predictions))

    confidence = float(predictions[predicted_index])

    if predicted_index < len(labels):
        predicted_label = labels[predicted_index]
    else:
        predicted_label = f"Class {predicted_index + 1}"

    return predicted_label, confidence


# ---------------------------------
# MAIN APPLICATION
# ---------------------------------

st.title("♻️ AI Smart Waste Segregation")
st.subheader("Waste Helper & Recycling Assistant")

st.write(
    "Upload an image of waste to identify its category "
    "and learn how it can be handled."
)

# Load model and labels
try:
    model = load_waste_model()
    labels = load_labels()

except Exception as error:
    st.error("Unable to load the AI model or labels.")
    st.code(str(error))
    st.stop()


# ---------------------------------
# IMAGE UPLOAD
# ---------------------------------

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Waste Image",
        use_container_width=True
    )

    if st.button("Identify Waste", type="primary"):

        with st.spinner("Analyzing image..."):

            try:
                category, confidence = predict_waste(
                    image,
                    model,
                    labels
                )

                st.success("Waste identification completed!")

                st.subheader("Prediction Result")

                st.write("**Waste Category:**", category)

                st.write(
                    f"**Confidence:** {confidence * 100:.2f}%"
                )

                st.progress(
                    max(0.0, min(1.0, confidence))
                )

                # General guidance based on prediction
                category_lower = category.lower()

                st.subheader("♻️ Recycling Guidance")

                if any(word in category_lower for word in
                       ["plastic", "bottle"]):

                    st.info(
                        "Plastic: Separate recyclable plastic "
                        "and follow local recycling instructions."
                    )

                elif any(word in category_lower for word in
                         ["paper", "cardboard"]):

                    st.info(
                        "Paper/Cardboard: Keep it dry and separate "
                        "it for paper recycling."
                    )

                elif any(word in category_lower for word in
                         ["metal", "can"]):

                    st.info(
                        "Metal: Separate metal items for recycling."
                    )

                elif any(word in category_lower for word in
                         ["glass"]):

                    st.info(
                        "Glass: Keep glass separate and handle "
                        "broken glass safely."
                    )

                elif any(word in category_lower for word in
                         ["organic", "food"]):

                    st.info(
                        "Organic Waste: Suitable food waste may "
                        "be collected for composting."
                    )

                else:

                    st.info(
                        "Check your local waste collection rules "
                        "for the identified category."
                    )

            except Exception as error:

                st.error("Prediction failed.")

                st.code(str(error))


# ---------------------------------
# FOOTER
# ---------------------------------

st.divider()

st.caption(
    "AI Smart Waste Helper | Waste Segregation "
    "& Recycling Assistant"
)
