
###############################################################################
from struct import pack_into
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
from skimage import io
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
import cv2
import glob
import os.path
import ipywidgets as widgets
#print("Mark your wall pattern")
import glob
#from __future__ import print_function
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import shutil
import os



############################################################################################
############################################################################################

#st.title('Pattern Recognition')

st.markdown("<h1 style='text-align: center; color: red;'>Pattern Recognition</h1>", unsafe_allow_html=True)
st.write("These Patterns are identified in your drawings")   


# images = []
# for img_path in glob.glob('D:/Winjit/Training/Assignment/Construction_Use_Cases/automate_2D_qto/floorplan_element_det/images/subplt_temp/*.jpg'):
#      images.append(mpimg.imread(img_path))

images = []



path_temp = 'C:/Users/ChinmayB/Desktop/location/Patterns'
for img_path in glob.glob(path_temp+'\\'+'*.jpg'):
     images.append(mpimg.imread(img_path))


# plt.figure(figsize=(20,10))
# columns = 5
for i, image in enumerate(images):
#     plt.subplot(len(images) / columns + 1, columns, i + 1)
    #st.image(image, width=280)
    with st.container():
        for col in st.columns(4):
            col.image(image, width=50)
        st.subheader("pattern_number_{}".format(i))

st.text("----------------------------------------------------------------------------------------------------")



st.subheader("Please Choose your wall pattern")   

    # plt.imshow(image)
    #plt.text(25,150,"pattern {}".format(i),va = 'top')
    #plt.title("Pattern_number {}".format(i))
   # fig.text(1, 5, 'Pattern {}',format(i), ha='left', va='top')


################################################################################
#############################  Selection of pattern ###########################
####################################################################################


option = st.selectbox(
     'Select here:',
    ["None","Pattern_number_0", "Pattern_number_1", "Pattern_number_2","Pattern_number_3","Pattern_number_4","My pattern is not matching "])
st.write('You selected:', option)


ww = []
# src_dir = r'D:\Winjit\Training\Assignment\Construction_Use_Cases\automate_2D_qto\floorplan_element_det\images\subplt_temp'
path_temp_sel = r'C:\Users\ChinmayB\Desktop\location\selected_pattern_for_element'

for filename in os.listdir(path_temp_sel):
    file_path = os.path.join(path_temp_sel, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))


#User_input = input("Give_Cilck_input ")  
user_response = option

    

for jpgfile in glob.glob(os.path.join(path_temp, "*.jpg")):   
    #  jpgfile.split("\n")
    ww.append(jpgfile)
    ww.sort()
# if user_response == "None":
#     print("No Option is selected")
#     pass
if user_response == "Pattern_number_0":
        shutil.copy(ww[0], path_temp_sel)
            
if user_response == "Pattern_number_1":
        shutil.copy(ww[1], path_temp_sel)
        templete = ww[1]
if user_response == "Pattern_number_2":
        shutil.copy(ww[2], path_temp_sel)
if user_response == "Pattern_number_3":
        shutil.copy(ww[3], path_temp_sel)
if user_response == "Pattern_number_4":
        shutil.copy(ww[4], path_temp_sel)
if user_response == "I have different pattern":
    pass    
st.write("Template identified successfully ")   

for x in range(1):
    if user_response == 'None':   
        print("No Option is selected")
        continue

    else:        
        #####################################################################################
        ############################ code to rotate templetes ##############################
        ####################################################################################

        # datagen = ImageDataGenerator(horizontal_flip=True)

        # image_directory = 'C:/Users/ChinmayB/Desktop/location/selected_pattern_for_element/'
        # SIZE = 209
        # dataset = []

        # my_images = os.listdir(image_directory)
        # for i, image_name in enumerate(my_images):
        #     if (image_name.split('.')[1] == 'jpg'):
        #         image = io.imread(image_directory + image_name)
        #         image = Image.fromarray(image,'RGB')
        #         image = image.resize((SIZE,SIZE))
        #         dataset.append(np.array(image))
                
        # x = np.array(dataset)

        # i = 0
        # for batch in datagen.flow(x, batch_size=1,
        #                           save_to_dir='C:/Users/ChinmayB/Desktop/location/selected_pattern_for_element/', save_prefix='Element', save_format='jpeg'):
        #     i += 1
        #     if i > 2:
        #         break

        ##################################################################################
        ###################################### FATEMA'S CODE ###########################

        # temp_dir = r'D:\Winjit\Training\Assignment\Construction_Use_Cases\automate_2D_qto\floorplan_element_det\images\pattern_match\selected_temp'
        for file in os.listdir(path_temp_sel):            
            fname,ext = os.path.splitext(file)

            if ext == '.jpg':
                print(file)
                template = cv2.imread(path_temp_sel+'\\'+file)
                imgrot = cv2.rotate(template,cv2.ROTATE_90_CLOCKWISE)
                # cv2.imshow("rotated",imgrot)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(path_temp_sel+'\\'+'rotated_{0}.jpg'.format(fname),imgrot)
        #################################################################################
        #################################################################################
        ################################# ORIGINAL #################################

        
        #######################################################################################


        # res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        # plt.imshow(res, cmap='gray')
        #####################################################################
        
            
        points = []
        img_rgb = cv2.imread(r'C:\Users\ChinmayB\Desktop\location\test-601.png')
        #img = cv2.resize(img_rgb, (960, 1000))  
        img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)

        # temp_dir = r'D:\Winjit\Training\Assignment\Construction_Use_Cases\automate_2D_qto\floorplan_element_det\images\pattern_match\rotated'
        lbl = ['column','wall']
        k = 0
        for filename in os.listdir(path_temp_sel):
            print(filename)
            template = cv2.imread(path_temp_sel+'\\'+filename)
            temp_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        #     print(temp_dir+'\\'+filename)
        #     print(template.shape)
        #     cv2.imwrite("%s.jpg" % filename ,frame)
            h,w = temp_gray.shape[:]
            
            res = cv2.matchTemplate(img_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
            plt.imshow(res, cmap = 'gray')
            #threshold = 0.70
            threshold = st.slider('select threshold', 0, 100, 10)
            loc = np.where(res >= threshold)
            
            
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0),-1)
            print(w,h)
            print(lbl[k])
            cv2.imwrite(r"C:\Users\ChinmayB\Desktop\location\Match_image_{0}.png".format(lbl[k]), img_rgb)
            k+=1

        #####################################################################

        image2 = Image.open(r'C:\Users\ChinmayB\Desktop\location\Match_image_wall.png')        
        # image3 = cv2.imread(r"D:\Winjit\Training\Assignment\Construction_Use_Cases\automate_2D_qto\floorplan_element_det\images\pattern_match\out.jpg")
        st.subheader("Matched Pattern in image")
        st.image(image2,caption = "Possible simillar patterns in floor plan")
        
        ####################################################################################

        def cropping_cor(img):
            x,y = img.shape[0:2]
            print("X is",x)
            print("Y is ",y)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # convert the grayscale image to binary image

            ret,thresh = cv2.threshold(gray_image,127,255,0)
            # calculate moments of binary image

            M = cv2.moments(thresh)
            # calculate x,y coordinate of center

            cX = int(M["m10"] / M["m00"])

            cY = int(M["m01"] / M["m00"])
            # put text and highlight the center

            cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
            h = int(cX * 0.9)
            w = int(cY * 0.3)
            n = int(cY * 0.9)
            cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # display the image
            print("the centre coordinates are cX,cY:",cX,cY)
        #     cv2.line(img, (cX,cY), (cX,h), (255,0,0), 2) 
        #     cv2.line(img,(cX,cY),(cX,cX-h),(255,0,0),2)
        #     cv2.line(img,(cX,cY),(w,cY),(255,0,0),2)
        #     cv2.line(img,(cX,cY),(y-w,cY),(255,0,0),2)
            # # #final_horizontal = cv2.line(img,(458,0),(458,916),(255,0,0),2)
            #final_vertical = cv2.line(final_horizontal,(0,224),(916,224),(255,0,0),2)
            #cv2.line(img, (458 ,224), (0,y), (255,0,0), 2)
        #     cv2.line(img,(cX,h),(cX,cX-h),(255,0,0), 2)
        #     cv2.line(img,(w,cY),(y-w,cY),(255,0,0), 2)
            cv2.line(img,(w,cX-h),(y-w,cX-h),(255,0,0), 2)
            cv2.line(img,(w,2*n),(y-w,2*n),(255,0,0), 2)

            cord_1,cord_2 = w,cX-h
            cord_3,cord_4 = y-w,cX-h
            cord_5,cord_6 = w,2*n
            cord_7,cord_8 = y-w,2*n
            print("First coordinates:",cord_1,cord_2)
            print("second coordinates:",cord_3,cord_4)
            print("third coordinates:",cord_5,cord_6)
            print("fourth coordinates:",cord_7,cord_8)

            

            crop_img2 = img[cord_2:cord_6,cord_1:cord_3]
            cv2.imwrite("C:/Users/ChinmayB/Desktop/Images datasets/cropped_img2.jpg",crop_img2)




        img = cv2.imread(r'C:\Users\ChinmayB\Desktop\location\Match_image_wall.png')
        cropping_cor(img)
            


        #####################################################################################
  ############################## optional mouse crop ################################################
  ################################################################################################
       



# img = cv2.imread(r'C:\Users\ChinmayB\Desktop\location\Match_image_wall.png')
# img = cv2.resize(img, (630, 600))
# j = 0
# for i in range(1):
#     cv2.namedWindow('image_{0}'.format(i))
#     cv2.setMouseCallback('image_{0}'.format(i), mouse_crop)    
#     cv2.imshow('image_{0}'.format(i), img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     j+=1
            
        ################################################################################
        ######################### Measurment of color Pixel ##############################
        #################################################################################
        image4 = cv2.imread("C:/Users/ChinmayB/Desktop/Images datasets/cropped_img2.jpg")
        # im = Image.open(r"C:\Users\ChinmayB\Desktop\location\out.jpg")
        # im.save(r"C:\Users\ChinmayB\Desktop\location\test-600.png", dpi=(150,150))
        #cv2.imshow('image3', image3)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #image3 = cv2.imread(r"C:\Users\ChinmayB\Desktop\location\test-600.png")


        image3 = Image.open("C:/Users/ChinmayB/Desktop/Images datasets/cropped_img2.jpg")
        st.image(image3,caption = "Final pattern matching ")

                    # counting the number of pixels
        number_of_color_pix = np.sum(image4 == (0, 255, 0))
        print(number_of_color_pix)
                
                #number_of_black_pix = np.sum(img == 0)
        total_area = number_of_color_pix * 8.35
        total_area = round(total_area,2)

        st.header("Total area of columns is {} mm-square".format(total_area))
                
        st.header('Number of Color pixels: {}'. format( number_of_color_pix))
        print(("Total area of columns is {} mm-square".format(total_area)))

                

#####################################################################################
####################################################################################
        option = st.selectbox(
        'Select here:',
        ["Satisfied by results", "want to crop mannually"])
        st.write('You selected:', option)
        if option == 'want to crop mannually':
            def mouse_crop(event, x, y, flags, param):
                        
                # grab references to the global variables
                global x_start, y_start, x_end, y_end, crop1, crop2, j      
            #     j+=1

                #lbl = ['column','wall']
            #     print(j)
            #     print(lbl[j])
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    x_start, y_start = x, y
                    crop1= True

                elif event == cv2.EVENT_LBUTTONUP:
                
                    x_end, y_end = x, y
                    crop2 = True
                    cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (255, 255, 0), 3)
                    cv2.imshow('input_{0}'.format(i), img)
            #         lbl = ['column','wall']
                    
                
                    if crop1== True and crop2 == True:
                        refPoint = [(x_start, y_start), (x_end, y_end)]
                        roi = img[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                        
                        cv2.imwrite(r"C:\Users\ChinmayB\Desktop\location\out.jpg",roi)
                        print('done')
                    
                        cv2.imshow("Cropped_{0}".format(i), roi)
                
                crop1 = False
                crop2= False

                


            img = cv2.imread(r'C:\Users\ChinmayB\Desktop\location\Match_image_wall.png')
                

            img = cv2.resize(img, (630, 600))       
            j = 0
            for i in range(1):
                

                cv2.namedWindow('image_{0}'.format(i))
                cv2.setMouseCallback('image_{0}'.format(i), mouse_crop)    
                cv2.imshow('image_{0}'.format(i), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                j+=1
            img_crop_mouse = Image.open(r"C:\Users\ChinmayB\Desktop\location\out.jpg")
            st.image(img_crop_mouse,caption = "Your result by mouse cropping ")
                      # counting the number of pixels
            number_of_color_pix = np.sum(img_crop_mouse == (0, 255, 0))
            print(number_of_color_pix)
                    
                    #number_of_black_pix = np.sum(img == 0)
            total_area = number_of_color_pix * 8.35
            total_area = round(total_area,2)

            st.header("Total area of columns is {} mm-square".format(total_area))
                    
            st.header('Number of Color pixels: {}'. format( number_of_color_pix))
            print(("Total area of columns is {} mm-square".format(total_area)))

                            
                            
        



                        
