import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
html_string = "<h3 style='color:blue;'>SOLAR RADIATION PREDICTION..</h3>"

st.markdown(html_string, unsafe_allow_html=True)

#st.markdown("<span style=“background-color:#121922”>",unsafe_allow_html=True)




image=Image.open("solar2.jpg")
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
    fever =st.sidebar.slider("Time", 0, 17, 3)
    headache=st.sidebar.slider("radiation", 0, 199, 117)
    cough=st.sidebar.slider("humidity", 0, 122, 72)
    shortness_of_breathe=st.sidebar.slider("temperature", 0, 99, 23)
    insulin=st.sidebar.slider("pressure", 0.0, 846.0, 30.5)
    soarthroat=st.sidebar.slider("windspeed", 0.0,67.1, 32.0)
    DPF=st.sidebar.slider("sunrise", 0.078, 2.42, 0.3725)
    age=st.sidebar.slider("sunset", 21, 81, 29)
    
    
    user_data={"fever": Fever,
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
    

RandomForestClassifier=RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)



st.subheader("Model Test Accuracy:")
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test)) *100)+'%' )


pediction=RandomForestClassifier.predict(user_input)


st.subheader("Classification:")
st.write(pediction)
if pediction==0:
    st.success("You are Not Suffering from COVID-19")
else:
    st.error("Oops..!! You are suffering from COVID=19")


checkbox6=st.sidebar.checkbox("COVID CASES ACROSS THE WORLD")

if checkbox6:
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




checkbox1=st.sidebar.checkbox("active_cases")

if checkbox1:
    st.subheader("Active cases:")
    st.bar_chart(df.active_cases)


checkbox2=st.sidebar.checkbox("death_cases")
if checkbox2:
    st.subheader("Death cases:")
    st.bar_chart(df.death_cases)

checkbox3=st.sidebar.checkbox("recovered_cases")

if checkbox3:
    st.subheader("Recovered  cases:")
    st.bar_chart(df.recovered_cases)

checkbox4=st.sidebar.checkbox("confirmed_cases")

if checkbox4:
    st.subheader("Confirmed cases:")
    st.bar_chart(df.confirmed_cases)


