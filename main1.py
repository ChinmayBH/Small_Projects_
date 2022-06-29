import sys
import cv2
import sys
import streamlit as st
import cv2

sys.path.append('./TF2DeepFloorplan/')
sys.path.append('./TF2DeepFloorplan/dfp')
from dfp.net import *
from dfp.data import *
import matplotlib.image as mpimg
from argparse import Namespace

sys.path.append('./TF2DeepFloorplan/dfp/utils/')
from dfp.utils.rgb_ind_convertor import *
from dfp.utils.util import *
from dfp.utils.legend import *
from dfp.deploy import *


#input image
def seg(inp):
        inp1 = mpimg.imread(inp)

        args = Namespace(image=inp1,
                weight='./log/log/store/G',loadmethod='log',
                postprocess=True,colorize=True,
                save=None)
        result = main(args)
        st.image(result)



        # print(type(result))
        # cv2.imwrite("save.jpg",result)

# plt.subplot(1,2,1)
# plt.imshow(inp); plt.xticks([]); plt.yticks([]);
# plt.subplot(1,2,2)
# plt.imshow(result); plt.xticks([]); plt.yticks([]);

