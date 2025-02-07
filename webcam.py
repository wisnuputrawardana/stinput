import streamlit as st
from PIL import Image
from simple_facerec import SimpleFacerec
import cv2
import os
import numpy as np

root = os.getcwd()

fr = SimpleFacerec()
fr.load_encoding_images(f"{root}/db")
st.title("Select Camera and Capture Image")

# Let users capture an image using their camera
image_file = st.camera_input("Take a picture", disabled=False)
placeholdertext = st.empty()
if image_file:
    # Convert to PIL Image and display
    image = Image.open(image_file)
    image.save(f"{root}/detected/captured_image.jpg")
    np_img = np.array(image)
    faceLoc, faceName = fr.detect_known_faces(np_img)
    for face_loc, name in zip(faceLoc, faceName):
        y, x, h, w = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(np_img,name,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0), 2)
        cv2.rectangle(np_img, (x, y), (w, h), (255, 0, 0), 2)
    if faceName:
        name = faceName[0]
        placeholdertext.write(name)
    else:
        name = "Unknown"
        placeholdertext.write("Unknown")
    
    st.image(np_img, caption="Captured Image")