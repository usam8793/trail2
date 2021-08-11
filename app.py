# streamlit 
import streamlit as st

import pandas as pd
import numpy as np
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import pickle


def load_models(name):

    if name =='K-Nearest Neighbours':

        pickle_in = open("model_knn", "rb")
        classifier = pickle.load(pickle_in)
        return classifier

    elif name == 'SVM':
        
        pickle_in = open("model_svm", "rb")
        classifier = pickle.load(pickle_in)
        return classifier



# feat_data
feat_data = None


st.markdown('<h1 style="text-align:center;"><u>Diabetes Classifier</u></h1>',unsafe_allow_html=True)

st.markdown('***')



with st.form(key='my_form'):
    st.markdown('<center>Medical Survey</center>',unsafe_allow_html=True)
    n_preg = st.number_input('Number of Pregnancies', min_value=0, max_value=20,key='preg' ,value=1)
    glucose = st.number_input("Glocose level", min_value=0, max_value=200, key='glucose',value=85)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=122, key='bp',value=66)
    skinthickness = st.number_input('Skin Thickness',min_value=0, max_value=100, step=5, key='skin',value=29)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, key='insulin',value=0)
    bmi = st.number_input("BMI",max_value=70.00,step=0.1,key='bmi',value=26.6)
    dpf = st.number_input('Diabetes Pedigree Function Value', min_value=0.078, max_value=2.420,step=0.1,key='dpf',value=0.351)
    age = st.number_input('Age', min_value=21, max_value=90, step=5, key='age',value=31)

    submit_button = st.form_submit_button(label='Predict')


if submit_button:       
    feat_data = [n_preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    
    st.markdown('Feature Data collected.')
    st.code(feat_data)

# slot 
slot = st.empty()
    

st.markdown('> <h2><b>Prediction</b></h2>',unsafe_allow_html=True)

class_index = {0: "Positive", 1: "Negative"}

model_name = st.selectbox("Select a Machine Learning Model", ['K-Nearest Neighbours', 'SVM'], key='selectml')

classifier = load_models(name=model_name)


if not feat_data == None:
    st.markdown(f"***{model_name}: Outcome Prediction ***")


    prediction = classifier.predict([feat_data])
    # st.markdown(prediction)

    if prediction[0] == 0:
        st.success(f'***{class_index[prediction[0]]}*** for Diabetes')
    else:
        st.success(f'***{class_index[prediction[0]]}*** for Diabetes')

else:
    slot.warning('***Complete Filling the Form and Submit***')



st.markdown("***")





