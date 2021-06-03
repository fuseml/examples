#To use the app install first "streamli" - https://docs.streamlit.io/en/stable/troubleshooting/clean-install.html
# Once installed run either:
# streamlit run <the raw URL of this app> or python -m streamlit run <the raw URL of this app>

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import requests

st.title("Winery Co. Quality App")
st.header("Welcome to the Winery Co. quality app. This app will help you to identify the quality of your wine.")
st.subheader("The app use a prediction model to show the results")


inferURL=st.text_input('Please enter the inference URL as printed by the FuseML workflow output')
st.write("Once your URL is setup, use the sliders on the left to set your request values, then hit the predict button below..")



# Add sidebar options
st.sidebar.title("Select your wine combination")
st.sidebar.header("Once selected please hit the predict button in the center of the page")
option1= st.sidebar.slider('Alcohol', min_value=8.00, max_value=14.90)
option2= st.sidebar.slider('Chlorides', min_value=.000, max_value=1.000)
option3= st.sidebar.slider('Citric Acid', min_value=.00, max_value=1.00)
option4= st.sidebar.slider('Density', min_value=.00000, max_value=1.00000)
option5= st.sidebar.slider('Fixed Acidity', min_value=.0, max_value=16.0)
option6= st.sidebar.slider('Free Sulfur Dioxide', min_value=0, max_value=100)
option7= st.sidebar.slider('pH', min_value=2.0, max_value=5.0)
option8= st.sidebar.slider('Residual Sugar', min_value=.0, max_value=16.0)
option9= st.sidebar.slider('Sulphates', min_value=.00, max_value=2.00)
option10= st.sidebar.slider('Total Sulfur Dioxide', min_value=1, max_value=300)
option11= st.sidebar.slider('Volatile Acidity', min_value=.00, max_value=2.00)

#Central page predictions
def results():
    if inferURL == "":
         st.write("<h1 style='color:red;'>Please fill the Inference URL value above before trying to make a prediction</h1>",unsafe_allow_html= True)
    else:
        st.markdown('## Your wine selection is:')
        st.write(option1,option2,option3,option4,option5,option6,option7,option8,option9,option10,option11)
        st.markdown('<span/>',unsafe_allow_html = True)
        st.markdown('Your Wine quality is:',unsafe_allow_html = True)
        headers = {
            'Content-Type': 'application/json; format=pandas-split',
        }

        #data = '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[['+ str(option1) +','+ str(option2) +','+ str(option3) +','+ str(option4) +','+ str(option5) +','+ str(option6) +','+ str(option7) +','+ str(option8) +','+ str(option9) +','+ str(option10) +','+ str(option11) +']]}'
        data = '{"inputs": [{"name": "input-0","shape": [1, 11],"datatype": "FP32","data": [['+ str(option1) +','+ str(option2) +','+ str(option3) +','+ str(option4) +','+ str(option5) +','+ str(option6) +','+ str(option7) +','+ str(option8) +','+ str(option9) +','+ str(option10) +','+ str(option11) +']]}]}'

        # Un-comment the below line but change the <IP ADDR> to match your current environment IP
        #response = requests.post('http://<APPNAME>-workspace.<IP ADDR>.omg.howdoi.website/invocations', headers=headers, data=data)
        response = requests.post(inferURL, headers=headers, data=data)
        try:
            result = response.json().get('outputs')
            st.write("<h1>"+ str(result[0]['data'])+"</h1>",unsafe_allow_html= True)
        except:
            st.write("<h1 style='color:red;'>An exception occurred. Model is not ready yet</h1>",unsafe_allow_html= True)
        
        return

if st.button('Predict'):
    results()
