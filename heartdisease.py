import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
import seaborn as sns
from sklearn.cluster import KMeans
import  numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go




st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)


#st.markdown('<style>body{background-color: light pink;}</style>',unsafe_allow_html=True)
#st.markdown(""" <style> body { color: #fff; background-color: #111; } </style> """, unsafe_allow_html=True)
html_string = "<h3 style='color:blue;'>HEART FAILURE PREDICTION..</h3>"

st.markdown(html_string, unsafe_allow_html=True)

#st.markdown("<span style=“background-color:#121922”>",unsafe_allow_html=True)

st.write("""BEAT THE HEART DISEASE AND FEEL THE HEALTHY BEAT...
""")

image=Image.open("heart18.jpg")
st.image(image, caption='The most deadly disease is the failure of the heart...',use_column_width=True)

df=pd.read_csv("heart1.csv")







st.subheader("Data Information:")


st.dataframe(df)
st.write(df.describe())
a=pd.read_csv("heart1.csv")
sns.heatmap(a.isnull())



st.subheader("Data Prediction:")
chart=st.bar_chart(df)

X=df.iloc[:,0:13].values
Y=df.iloc[:,-1].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


def get_user_input():
    #pregnancies=st.sidebar.slider.markdown("<h1 style='color: Blue;'>'pregnancies', 0, 17, 3</h1>",unsafe_allow_html=True)
    Age =st.sidebar.number_input("Age",20, 200,20)
    sex=st.sidebar.number_input("Sex(0=Male,1=Female)", 0, 1,0)
    cp=st.sidebar.number_input("Cp (Chest pain)", 0, 10, 0)
    trestbps=st.sidebar.number_input("Trestbps(Resting Blood pressure(in mm Hg))", 0, 900, 100)
    chol=st.sidebar.number_input("Chol(serum cholestral in mg/dl)", 0, 846, 110)
    fbs=st.sidebar.number_input("FBS(Fasting blood Sugar>120 mg/dl)(1=True,0=False)", 0,1, 0)
    restecg=st.sidebar.number_input("Restecg(Resting electrocardiographic results)", 0, 1, 0)
    thalach=st.sidebar.number_input("Thalach(maximun heart rate achieved)", 100, 800, 100)
    exang=st.sidebar.number_input("Exang(exercise induced angia (1=Yes,0=No))", 0, 1, 0)
    oldpeak=st.sidebar.number_input("Oldpeak(ST depression induced by exercise relatedto rest)", 0.0, 10.0, 0.0)
    slope=st.sidebar.number_input("Slope(The Slope of the peak exercise ST segment)", 0, 5, 1)
    ca=st.sidebar.number_input("Ca(No of major vesseles(0-3)colored by flurosopy)", 0, 5, 1)
    thal=st.sidebar.number_input("Thal(3=normal,6=fixed detect,7=reversable detect)", 0, 5, 3)
    

    
    
    
    
    
    
    
    user_data={"Age": Age,
              "sex": sex,
              "cp":cp,
              "trestbps":trestbps,
              "chol" :chol,
              "fbs" :fbs,
              "restecg" :restecg,
              "thalach" :thalach,
              "exang" :exang,
              "oldpeak" :oldpeak,
              "slope" :slope,
              "ca" :ca,
              "thal" :thal}
    
                
    features=pd.DataFrame(user_data,index=[0])
    return features


user_input=get_user_input()

st.subheader("User Input Data:")
st.write(user_input)

    

clf=RandomForestClassifier()
clf.fit(X_train,Y_train)



st.subheader("Model Test Accuracy:")
st.write(str(accuracy_score(Y_test,clf.predict(X_test)) *100)+'%' )


pediction=clf.predict(user_input)


st.subheader("Classification:")
st.write(pediction)




if pediction==0:
    st.success("You are Healthy...!!!!")
else:
    st.error("Oops..!! You are not Healthy...!!!!")

st.subheader("Data Visualization using seaborn:")
image=Image.open("seaborn.jpg")

st.image(image,width=500)

st.subheader("Data Visualization using Pyplot:")
image=Image.open("seaborn2.jpg")

st.image(image,width=800,cmap="dark12")
st.subheader("Prediction of the Data:")
image1=Image.open("sex.jpg")
image2=Image.open("cp.jpg")
image3=Image.open("trestbps.jpg")
image4=Image.open("chol.jpg")
image5=Image.open("fbs.jpg")
image6=Image.open("restecg.jpg")
image7=Image.open("thalach.jpg")
image8=Image.open("exang.jpg")
image9=Image.open("oldpeak.jpg")
image10=Image.open("slope.jpg")
image11=Image.open("ca.jpg")
image12=Image.open("thal.jpg")
st.image([image1,image2,image3,image4,image5,image6,image7,image8,image9,image10,image11,image12],width=300)


st.subheader("Leading Cases of Deaths:")
if st.button("Click Here"):
    labels = ['Cancer','Chronic and Respiratory_Diseases','Heart Disease','Accidents','Stroke','Aizheimer_Disease','Diabetes','Flu_pneumonia','Kidney_Disease','Suicide']
    values = [595930,155041,633842,146571,140323,110561,79535,57062,49959,44193]

    # pull is given as a fraction of the pie radius 
    fig = go.Figure(data=[go.Pie(labels=labels, values=values,title='LEADING CASES OF DEATHS' ,pull=[0, 0, 0.2, 0])])
    fig.show()



st.subheader("Ages Affected:")
if st.button("Click Here.."):
    labels = ['Babies','Toddlers','Childrens','Teemagers','Young_Adults','Adults','Seniors']
    values = [2,5,13,18,40,60,70]

    fig = go.Figure(data=[go.Pie(labels=labels,title='AGES AFFECTED',values=values)])
    fig.show()

st.subheader("Future of CVD:")
if st.button("Click Here."):
    labels = ['2021','2022','2023','2024','2025','2026','2027','2028','2029','2030']
    values = [2300000,2400000,2500000,2600000,2734067,2809876,2908765,3000000,3098765,3198765,3298765]
    fig = go.Figure(data=[go.Pie(labels=labels,title='Future of CVD', values=values,pull=[0, 0, 0,0,0,0,0,0,0,0.2])])
    fig.show()
