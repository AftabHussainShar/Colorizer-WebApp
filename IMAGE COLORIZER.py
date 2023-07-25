import streamlit as st
import numpy as np
from PIL import Image
import torch
from io import BytesIO

from colorizers import *

# Set page configuration
st.set_page_config(page_title='IMAGE COLORIZER', page_icon='path_to_logo.png')

# Custom CSS styling
st.markdown(
    """
    <style>
    .header-text {
        font-size: 48px;
        background: -webkit-linear-gradient(left, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
        animation: fadeInFromLeft 1s ease-in-out;
    }
    .subheader-text {
        font-size: 40px;
        background: -webkit-linear-gradient(left, #FF512F, #F09819);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        animation: fadeInFromRight 1s ease-in-out;
    }
    .title-text {
        font-size: 72px;
        background: -webkit-linear-gradient(left, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    @media only screen and (max-width: 768px) {
        .title-text {
            font-size: 60px;
        }
    }
    .colorized-container {
        text-align: center;
        margin-bottom: 20px;
    }
    .colorized-image {
        border: 2px solid #00C9FF;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .input-container {
        text-align: center;
    }
    .input-image {
        border: 2px solid #92FE9D;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cache the model
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if torch.cuda.is_available():
    colorizer_siggraph17.cuda()

@st.cache(persist=True, allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def process_image(img):
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    if torch.cuda.is_available():
        tens_l_rs = tens_l_rs.cuda()
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    return out_img_siggraph17

def fix_channels(img):
    if len(img.shape) == 2:
        return np.dstack((img, img, img))
    else:
        return img

# Header section
st.markdown("<p class='header-text'>IMAGE COLORIZER</p>", unsafe_allow_html=True)

# Subheader section
st.markdown("<p class='subheader-text'>Add colors to your black-and-white images!</p>", unsafe_allow_html=True)

# Upload image or submit
st.subheader('Please upload an image.')

uploaded_file = None

with st.form(key='uploader'):
    uploaded_file = st.file_uploader("Choose a file...")
    submit_button_upl = st.form_submit_button(label='Submit image')

if uploaded_file and submit_button_upl:
    img = Image.open(uploaded_file)
    img = np.array(img)
    img = fix_channels(img)
    with st.spinner(f'Colorizing image, please wait...'):
        out_img = process_image(img)

        # Wrap the images in containers with custom CSS class
        st.markdown("<div class='colorized-container'><p class='colorized-image'>Colorized Image</p></div>", unsafe_allow_html=True)
        st.image(out_img, use_column_width=True)

        st.markdown("<div class='input-container'><p class='input-image'>Input Image</p></div>", unsafe_allow_html=True)
        st.image(img, use_column_width=True)

        # Convert the numpy array to bytes and add a download button for the colored image
        buffer = BytesIO()
        Image.fromarray(out_img.astype(np.uint8)).convert('RGB').save(buffer, format='PNG')
        st.download_button(
            label="Download Colorized Image",
            data=buffer.getvalue(),
            file_name="colorized_image.png",
            mime="image/png",
        )
