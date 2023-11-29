import streamlit as st
from PIL import Image 

st.title('Autocolorization App')
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Colorize the image using your model
    # colorized_image = autocolorize_image(image)

    # Display the colorized image
    # st.image(colorized_image, caption='Colorized Image', use_column_width=True)

