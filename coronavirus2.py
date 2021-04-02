import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from PIL import Image
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


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
html_string = "<h3 style='color:blue;'>CORONAVIRUS PREDICTION..</h3>"

st.markdown(html_string, unsafe_allow_html=True)

#st.markdown("<span style=“background-color:#121922”>",unsafe_allow_html=True)

st.write("""COVID-19 YOUR LIFE IS IN OUR HANDS...
""")



image=Image.open("image3.jpg")
st.image(image, caption='The safety of the people shall be the highest duty...',use_column_width=True)

df=pd.read_csv("coron.csv")







st.subheader("Data Information:")


st.dataframe(df)
st.write(df.describe())
a=pd.read_csv("coron.csv")
sns.heatmap(a.isnull())



st.subheader("Data Prediction:")
chart=st.bar_chart(df)

X=df.iloc[:, 0:8].values
Y=df.iloc[:,-1].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


def get_user_input():
    #pregnancies=st.sidebar.slider.markdown("<h1 style='color: Blue;'>'pregnancies', 0, 17, 3</h1>",unsafe_allow_html=True)
    fever =st.sidebar.slider("Fever", 0, 17, 3)
    headache=st.sidebar.slider("Headache", 0, 199, 117)
    cough=st.sidebar.slider("Cough", 0, 122, 72)
    shortness_of_breathe=st.sidebar.slider("Shortness_of_breathe", 0, 99, 23)
    insulin=st.sidebar.slider("Insulin", 0.0, 846.0, 30.5)
    soarthroat=st.sidebar.slider("Soarthroat", 0.0,67.1, 32.0)
    DPF=st.sidebar.slider("DPF", 0.078, 2.42, 0.3725)
    age=st.sidebar.slider("Age", 21, 81, 29)
    
    
    user_data={"fever": fever,
              "headache": headache,
              "cough":cough,
              "shortness_of_breathe":shortness_of_breathe,
              "insulin" :insulin,
              "soarthroat" :soarthroat,
              "DPF" :DPF,
              "age" :age}
    
                
    features=pd.DataFrame(user_data,index=[0])
    return features


user_input=get_user_input()

st.subheader("User Input Data:")
st.write(user_input)
    
clf=DecisionTreeClassifier()
clf.fit(X_train,Y_train)



st.subheader("Model Test Accuracy:")
st.write(str(accuracy_score(Y_test,clf.predict(X_test)) *100)+'%' )


pediction=clf.predict(user_input)


st.subheader("Classification:")
st.write(pediction)
if pediction==0:
    st.success("You are Not Suffering from COVID-19")
else:
    st.error("Oops..!! You are suffering from COVID-19")
st.subheader("Data Visualization:")
image=Image.open("data.jpg")

st.image(image,use_column_width=True)
st.subheader("Prediction of the Data:")
image1=Image.open("headache.jpg")
image2=Image.open("fever.jpg")
image3=Image.open("ourthroat.jpg")
image4=Image.open("cougn.jpg")
image5=Image.open("function.jpg")
image6=Image.open("insulin.jpg")
image7=Image.open("breathe.jpg")
image8=Image.open("age.jpg")
st.image([image1,image2,image3,image4,image5,image6,image7,image8],width=300)

checkbox6=st.sidebar.checkbox("COVID CASES ACROSS THE WORLD")

if checkbox6:
    st.subheader("Analyse the data across the world:")
    vid_file = open("covid.mp4","rb").read()
    st.video(vid_file)



sns.set_style('darkgrid')


html_string = "<h3 style='color:blue;'>COVID-19 CASES ACROSS INDIA</h3>"

st.markdown(html_string, unsafe_allow_html=True)
st.write("""ACTICE CASES,
         RECOVERED CASES,
         CONFORMIED CASES,
         DEATH CASES...""")

df=pd.read_csv("coviddataset.csv")

st.subheader("Data Information:")

st.dataframe(df)

st.subheader("Decription of the Data:")
st.write(df.describe())

st.subheader("Data Split:")
st.area_chart(df)

checkbox1=st.sidebar.checkbox("ACTIVE CASES")

if checkbox1:
    st.subheader("Active cases:")
    st.bar_chart(df.active_cases)


checkbox2=st.sidebar.checkbox("DEATH CASES")
if checkbox2:
    st.subheader("Death cases:")
    st.bar_chart(df.death_cases)

checkbox3=st.sidebar.checkbox("RECOVERED CASES")

if checkbox3:
    st.subheader("Recovered  cases:")
    st.bar_chart(df.recovered_cases)

checkbox4=st.sidebar.checkbox("CONFIRMED CASES")

if checkbox4:
    st.subheader("Confirmed cases:")
    st.bar_chart(df.confirmed_cases)

import streamlit as st
import pandas as pd
import numpy as np
df=pd.read_csv("https://raw.githubusercontent.com/dexplo/bar_chart_race/master/data/urban_pop.csv")
df=pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
columns=['lat', 'lon'])
st.map(df)
