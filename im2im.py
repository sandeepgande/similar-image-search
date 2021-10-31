
import streamlit as st
from PIL import Image
import numpy as np
from google_images_search import GoogleImagesSearch
import glob
from io import BytesIO
import tensorflow as tf
import requests
import config

st.write("""
# IMAGE to IMAGE(S) Search
""")

try:
    inception_net
except:
    inception_net = tf.keras.applications.InceptionV3()
    response = requests.get("https://git.io/JJkYN")
    labels = response.text.split("\n")


try:
    gis
except:
    gis = GoogleImagesSearch(config.DEVELOPER_key, config.CX)

uploaded_file = st.file_uploader(label='choose an image...', type=['png', 'jpg'])

my_bytes_io = BytesIO()

if uploaded_file:
    n = st.number_input(label='No. of output images',value=15)
    image = Image.open(uploaded_file)
    image = image.resize((299, 299))
    img_array = np.array(image)
    input_image = img_array.reshape((-1, 299, 299, 3))
    st.text('image preview')
    st.image(img_array)

    if st.button('Fetch similar images'):
        
        with st.spinner('Please wait...'):

            input_image = tf.keras.applications.inception_v3.preprocess_input(input_image)
            prediction = inception_net.predict(input_image).flatten()
            lab =  {labels[i]: float(prediction[i]) for i in range(1000)}
            label = max(zip(lab.values(), lab.keys()))[1]



            # Setting the search parameters to fetch similar images
            _search_params = {
                'q': label,
                'num': n,
                'safe': 'high',
                'fileType': 'jpg',
                'imgType': 'photo',
                'imgSize': 'MEDIUM',
                'imgColorType': 'color',
                'rights': 'cc_noncommercial'
            }

            # Search for similar images through google image search api
            gis.search(search_params=_search_params)
            IMG = []
        st.header('Search results :')
        col1, col2, col3 = st.columns(3)
        for i,image in enumerate(gis.results()):
            my_bytes_io.seek(0)
            raw_image_data = image.get_raw_data()
            image.copy_to(my_bytes_io, raw_image_data)
            image.copy_to(my_bytes_io)
            my_bytes_io.seek(0)

            with col1:
                if (i)%3 == 0:
                    st.image(Image.open(my_bytes_io))
            with col2:
                if (i+2)%3 == 0:
                    st.image(Image.open(my_bytes_io))
            with col3:
                if (i+1)%3 == 0:
                    st.image(Image.open(my_bytes_io))

        
