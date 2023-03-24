import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from dice_coef_loss import dice_coef_loss, dice_coef



# Define a function to load and preprocess the input image
def load_image(image_bytes):
    nparr = np.frombuffer(image_bytes.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Define a function to perform image processing using a pre-trained deep learning model
def process_image(img):
    model = tf.keras.models.load_model('C:/Users/rusba/LIVER/models/liver_model_final_resunet.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    processed_img = model.predict(img)
    processed_img = np.squeeze(processed_img, axis=0)
    processed_img = cv2.resize(processed_img, (256, 256))
    processed_img = np.clip(processed_img, 0, 1)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
    return processed_img

# Define the Streamlit app
def app():
    st.title("Liver Cancer Detection")

    # Allow user to upload an image
    st.set_option('deprecation.showfileUploaderEncoding', False)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # If an image is uploaded, display it and process it
    if uploaded_file is not None:
    # Load and preprocess the input image
        img = load_image(uploaded_file)
    
        # Display the original image
        st.image(img, caption="Original Image", use_column_width=True)
    
        # Perform image processing using the deep learning model
        processed_img = process_image(img)
    
        # Display the processed image
        st.image(processed_img, caption="Processed Image", use_column_width=True)
        
# Run the Streamlit app
if __name__ == '__main__':
    app()

    

    

