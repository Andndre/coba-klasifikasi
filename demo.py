import streamlit as st
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import datetime

model = keras.models.load_model("final_CNN_model.h5")
 
st.write("""
# Demo App Digit Classification
""")

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 6)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#eee")
bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = Image.fromarray(img)
    # Predict digit
    img = img.resize((28, 28))
    img = img.convert("L")
    img = keras.preprocessing.image.img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype("float32") / 255.0
    prediction = model.predict([img])
    predicted_digit = prediction.argmax(axis=1)[0]
    st.write("### Predicted Digit: ", predicted_digit)
    st.write("Last updated at", datetime.datetime.now())
