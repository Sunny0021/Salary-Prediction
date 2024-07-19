import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import streamlit as st
import joblib

data = pd.read_csv('Salary Data.csv')
model = joblib.load('salary_model.pkl')


st.markdown("<h1 style = 'color: #0071AD; text-align: center; font-family: helvetica'>SALARY PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #0F080A; text-align: center; font-family: monospace '>Built By IHEMEGBULEM GODSTIME</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com(6).png')

st.markdown("<h4 style = 'margin: -30px; color: #0F080A; text-align: center; font-family: helvetica '>Project Overview</h4>", unsafe_allow_html = True)

st.write("Developing a predictive model for salary estimation based on age, gender, education, job title, and experience. Aim to provide accurate insights for workforce planning and compensation strategies.")

st.markdown("<br>", unsafe_allow_html= True)

st.dataframe(data, use_container_width= True)

st.markdown("<br>", unsafe_allow_html= True)
st.subheader('Input Variables', divider = True)

age = st.number_input('Age', data['Age'].min(), data['Age'].max())
gender = st.selectbox('Gender', data['Gender'].unique())
education_level = st.selectbox('Education Level', data['Education Level'].unique())
job_title = st.selectbox('Job Title', data['Job Title'].unique())
years_of_experience = st.number_input('Years of Experience', data['Years of Experience'].min(), data['Years of Experience'].max())

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

st.markdown("<h4 style = 'margin: -30px; color: #0F080A; text-align: center; font-family: helvetica '>User Inputs</h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html= True)

inputs = pd.DataFrame()
inputs['Age'] = [age]
inputs['Gender'] = [gender]
inputs['Education Level'] = [education_level]
inputs['Job Title'] = [job_title]
inputs['Years of Experience'] = [years_of_experience]

st.dataframe(inputs, use_container_width= True)

# import the transformers 
gender_trans = joblib.load('gender_encode.pkl')
education_level_trans = joblib.load('education_level_encode.pkl')
job_title_trans = joblib.load('job_title_encode.pkl')

# transform the input variables 
inputs['Gender'] = gender_trans.transform(inputs[['Gender']])
inputs['Education Level'] = education_level_trans.transform(inputs[['Education Level']])
inputs['Job Title'] = job_title_trans.transform(inputs[['Job Title']])

st.markdown("<br>", unsafe_allow_html= True)

prediction_button = st.button('Predict Salary')
if prediction_button:
   predicted = model.predict(inputs)
   st.success(f'The salary predicted for your worker is {predicted[0].round(2)}')
















