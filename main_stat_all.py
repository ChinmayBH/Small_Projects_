import numpy as np
import statistics
import streamlit as st
st.header("App to calculate all central tendancies and std")
def All_stat(your_data):
    print("Press 1 for mean")
    print("Press 2 for median")
    print("Press 3 for mode")
    print("Press 4 for variance")
    print("Press 5 for standard deviation")
    a = st.radio(
     "What you want to find ?",
     ("none",'Mean', 'Mode', 'Median',"variance",'st. deviation'))   
    l = len(your_data)
    if a == 'Mean':
        su = (np.sum(your_data))/l
        su = np.round(su,2)
        st.write("The mean is",su)
    if a == 'Median':
        your_data.sort() 
        d2 = np.median(your_data)
        st.write("The median is",d2)
    if a == 'Mode':
        w = statistics.mode(your_data)
        st.write("The mode is",w)
    if a == "variance":
        e = statistics.variance(your_data)
        e = round(e,3)
        st.write("The variance is",e)
    if a == 'st. deviation':
        r =  statistics.stdev(your_data)
        r = round(r,2)
        st.write("standard deviation is ",r)








        
Q = [3,1,3,5,7,7,8,9,0]   
All_stat(Q)
