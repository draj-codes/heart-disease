import streamlit as st
import pickle
import pandas as pd
import random
import os
st.header("Heart Diesease Prediction Using Machine learning ")
data='''Project Objective
Heart Disease Prediction using Machine Learning
Heart disease prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment. Machine Learning offers accurate prediction capabilities, enhancing healthcare outcomes.
In this project, I analyzed a heart disease dataset with appropriate preprocessing. Multiple classification algorithms were implemented in Python using Scikit-learn and Keras to predict the presence of heart disease.

Algorithms Used:
- Logistic Regression
- Naive Bayes
- Support Vector Machine (Linear)
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost
- Artificial Neural Network (1 Hidden Layer, Keras)'''


# file_path = os.path.join(os.path.dirname(__file__), "heart_model.pkl")
file_path = os.path.join(os.getcwd(), "heart_model.pkl")

# file_path = os.path.join(os.path.dirname(__file__), "models", "heart_model.pkl")
with open(file_path, "rb") as f:
    chatgpt = pickle.load(f)


st.subheader(data)
# with open(r"C:\Users\divya\OneDrive\Documents\GitHub\projet\heat-disease\Heart_disease_pred.pkl",'rb') as f:
#     chatgpt=pickle.load(f)

st.image('https://t-shikuro.github.io/images/heart/heart.gif')
df=pd.read_csv(r'C:\Users\divya\OneDrive\Documents\GitHub\projet\heat-disease\heart.csv')

st.sidebar.header("Select feature to predict heart disease")
st.sidebar.image("https://tse4.mm.bing.net/th/id/OIP.7LA1z7w-drtQmnFmC0KBNAHaE7?cb=thfvnext&pid=ImgDet&w=201&h=134&c=7&o=7&rm=3")


all_values = []

for i in df.iloc[:,:-1]:
    min_value, max_value = df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

final_value = [all_values]

ans = chatgpt.predict(final_value)[0]

import time

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Heart Disease')

place = st.empty()
place.image('https://media1.tenor.com/m/LLlSFiqwJGMAAAAC/beating-heart-gif.gif',width = 200)

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

if ans == 0:
    body = f'No Heart Disease Detected'
    placeholder.empty()
    place.empty()
    st.success(body)
    progress_bar = st.progress(0)
else:
    body = 'Heart Disease Found'
    placeholder.empty()
    place.empty()
    st.warning(body)
    progress_bar = st.progress(0)