# Python In-built packages
from pathlib import Path
import PIL
import cv2


# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Hebarium Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Hebarium Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "What Do You Want To Detect?", ['Vegetarian Elements','Non-Vegetarian Elements'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection O
if model_type == 'Vegetarian Elements':
    model_path = Path(settings.VEG_MODEL)
else : model_path = Path(settings.NON_VEG_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence,
                                    classes= 0
                                    )
                # boxes = res[0].boxes
                print(res[0])
                image= res[0].orig_img
                boxes= res[0].boxes.xyxy
                plate_imgs= [image[int(y1):int(y2), int(x1):int(x2)] for x1,y1,x2,y2 in boxes]
                plate_numbers= [helper.extract_text(processed_plate_img) for processed_plate_img in plate_imgs]

                # # Plot the detected objects on the video frame
                # res_plotted = res[0].plot()
                for i, plate_number_conf in enumerate(plate_numbers):
                    cv2.putText(image, f'{plate_number_conf[0]}', (int(boxes[i][0])-15, int(boxes[i][1]-15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 20, 255), 2, cv2.LINE_AA)
                    image= cv2.rectangle(image, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), (255, 20, 255), 2)
                # res_plotted = res[0].plot()[:, :, ::-1]
                st.image(image, caption='Detected Image',
                            channels= 'BGR',
                            use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

else:
    st.error("Please select a valid source type!")
