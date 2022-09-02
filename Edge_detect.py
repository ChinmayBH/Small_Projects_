

import streamlit as st
import cv2
from PIL import Image
import os
import glob
import numpy as np
import shutil
from PIL import Image
import base64


image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
if  image_file is not None:
    original_image = Image.open(image_file)
    st.image(original_image, caption='Uploaded image', use_column_width=True)
    image11 = np.asarray(original_image)
    grey_img = cv2.cvtColor(image11, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(grey_img)  
    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    invertedblur = cv2.bitwise_not(blur)
    option1 = st.slider('minimum threshold', 10, 130, 5)
    # option1 = st.selectbox(
    #     'Select lower value of threshold ',
    #     (10,20,30,40,50,60,70,80,90,100))  
    option2 = st.slider('Maximum threshold', 100, 300, 5)
    # option2 = st.selectbox(
    #     'Select lower value of threshold ',
    #     (100,120,150,170,190,210,220,230,240,250,260,270,280,290,300))  
    thresh_1 = option1
    thresh_2 = option2
    edges = cv2.Canny(image=invertedblur, threshold1=thresh_1, threshold2=thresh_2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = original_image.copy()
# draw the contours on a copy of the original image
    cv2.drawContours(image11, contours, -1, (0, 255, 0), 4)
    # contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    # big_contour = max(contours, key=cv2.contourArea)
    # edges1 = cv2.drawContours(edges, [big_contour], 0, (255,255,255), cv2.FILLED)
    st.image(edges,caption='end result')




    # thresh = cv2.threshold(original_image, thresh_1, thresh_2, cv2.THRESH_BINARY)[1]
    # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    # big_contour = max(contours, key=cv2.contourArea)

    # result = np.zeros_like(original_image)
    # cv2.drawContours(result, [big_contour], 2, (25,25,25), cv2.FILLED)
    # st.image(result,caption = "result of masking ")


# img = st.file_uploader("upload floor plan", type=['png', 'jpg'] )
# # img = cv2.imread('C:/Users/ChinmayB/Downloads/free-modern-house-plans-pdf-free-2048x1449.jpg')
# img = Image.open(img)
# # img = cv2.imread(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
# sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
# sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
# sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

# thresh_1 = 290
# thresh_2 = 300
# edges = cv2.Canny(image=img_blur, threshold1=thresh_1, threshold2=thresh_2) # Canny Edge Detection
# filename = 'savedImage1.jpg'
# # cv2.imwrite(filename, edges)
# st.image(edges)

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
