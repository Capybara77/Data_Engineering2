import io
from PIL import Image
from ultralyticsplus import YOLO, render_result
import streamlit as st

path_to_model = 'best.pt'
model = YOLO(path_to_model)

def process_image(image_upload):
    img = Image.open(image_upload)
    results = model.predict(img)
    render = render_result(model=model, image=img, result=results[0])
    return render.image

st.title('Recagnise image')

image_upload = st.file_uploader("Download Image", type=["jpg", "png", "jpeg"])

if image_upload is not None:
    st.image(image_upload, caption='Downloaded Image', use_column_width=True)

    with st.spinner('Image processing...'):
        result_image = process_image(image_upload)

    st.image(result_image, caption='Processed Image', use_column_width=True)