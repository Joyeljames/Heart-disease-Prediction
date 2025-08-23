import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Heart disease predictor",layout="centered")
st.title("Heart Disease prediction app")
st.markdown("""
This app is trained machine learning model to predict a person has heart disease or not
""")


st.sidebar.header("enter patient data")

age=st.sidebar.slider("Age",18,100,40)
gender=st.sidebar.selectbox("Gender",["Male","Female"])
cp=st.sidebar.selectbox("ChestPainType",["ATA","NAP","ASY","TA"])
trestbp=st.sidebar.slider("RestingBP",80,200,120)
chol=st.sidebar.slider("Cholesterol",100,400,200)
fbs=st.sidebar.selectbox("FastingBS",[0,1])
maxhr=st.sidebar.slider("MaxHR",60,220,150)
exa=st.sidebar.selectbox(" ExerciseAngina",["YES","NO"])
oldp=st.sidebar.slider("Oldpeak",0.0,6.0,1.0)
st_sl=st.sidebar.selectbox("ST_Slope",["up","Flat","Down"])

#encode as in training
gender_val=1 if gender=="Male" else 0
cp_map={"ATA":0,"NAP":1,"ASY":2,"TA":3}
exang=1 if exa=="YES" else 0
st_slope_map={"up":0,"Flat":1,"Down":2}

input_data=np.array([[age,gender_val,cp_map[cp],trestbp,chol,fbs,maxhr,exang,oldp,st_slope_map[st_sl]]])

#load model
model=joblib.load("new5.joblib")


if st.sidebar.button("Predict"):
    prediction=model.predict(input_data)[0]

    if st.subheader("prediction result"):
        st.error("patient likely to have heart disease")
    else:
        st.success("patien unlikely to have heart disease")



