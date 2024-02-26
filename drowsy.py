import dlib 
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image
from drowsiness_predicter.drowsy_detector import DrowsyDetector

detector = DrowsyDetector()

st.set_page_config(page_title = "Drowsiness Detection System",page_icon = " eye", layout = "wide")


col1 , col2 = st.columns([70,30])
with col1 :
    st.title(':eyes: Drowsiness Detection System')

with col2 :
    launched = st.button('Launch')

    if(launched):
        detector.run()



I1 = Image.open('images/dt.jpg')

st.markdown('________')

Data = pd.read_csv("datasets/drowsidata.csv")

st.markdown('**The below bar graph shows that the count of accidents happend due to drousiness and what are the percentage of drowsiness accident happen in a year.**')


Accidents = px.bar(Data ,x = 'Drowsiness_Accident_Cases',
                  y ='Year',
                  color= 'Percentage_of_Drowsiness_Accident_Cases' ,orientation='h')
Accidents.update_layout(xaxis_range =[1100,1350])

st.plotly_chart(Accidents,use_container_width = True)


st.markdown('_____')

st.markdown('**It shows the number of deaths in a year due to drowsiness accidents .**')

Deaths_due_to_accidents = px.line(Data , x = 'Year',y = 'Deaths_of_Drowsiness_Accident_Cases')

st.plotly_chart(Deaths_due_to_accidents,use_container_width = True)