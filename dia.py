import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.en import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from PIL import Image
import streamlit as st
import seaborn as sns
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
html_string = "<h3 style='color:blue;'>DIABETES PREDICTION..</h3>"

st.markdown(html_string, unsafe_allow_html=True)

#st.markdown("<span style=“background-color:#121922”>",unsafe_allow_html=True)

st.write("""Being diagnosed with diabetes can be a very scary thing, and it can easily make your life stand still for a moment..
""")

image=Image.open("diaimage.png")
st.image(image, caption='Insulin is not a cure for diabetes; it is a treatment. It enables the diabetic to burn sufficient carbohydrates so that proteins and fats may be added to the diet in sufficient quantities to provide energy for the economic burdens of life....',use_column_width=True)

df=pd.read_csv("dia.csv")







st.subheader("Data Information:")


st.dataframe(df)
st.write(df.describe())
a=pd.read_csv("dia.csv")
sns.heatmap(a.isnull())
st.subheader("Data Prediction:")
st.line_chart(df)
#st.bar_chart(df)

#st.subheader("Data Prediction:")
#chart=st.bar_chart(df)

X=df.iloc[:,0:11].values
Y=df.iloc[:,-1].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


def get_user_input():
    #pregnancies=st.sidebar.slider.markdown("<h1 style='color: Blue;'>'pregnancies', 0, 17, 3</h1>",unsafe_allow_html=True)
    Glucose =st.sidebar.number_input("Glucose",0, 200,0)
    BloodPressure=st.sidebar.number_input("BloodPressure", 10, 110,20)
    SkinThickness=st.sidebar.number_input("SkinThickness", 10, 110, 20)
    Insulin=st.sidebar.number_input("Insulin", 0, 300, 0)
    BMI=st.sidebar.number_input("BMI", 0.0, 50.0, 0.0)
    DiabetesPedigreeFunction=st.sidebar.number_input("DiabetesPedigreeFunction", 0.0, 10.0, 0.4)
    Age=st.sidebar.number_input("Age", 20, 110, 20)
    Pregnancies=st.sidebar.number_input("Pregnancies", 0, 15, 0)
    smoking=st.sidebar.radio("Smoking", (0, 1))
    anaemia=st.sidebar.radio("anaemia", (0, 1))
    Sex=st.sidebar.radio("Sex", (0, 1))
    
    

    
    
    
    
    
    
    
    user_data={"Glucose": Glucose,
              "BloodPressure": BloodPressure,
              "SkinThickness":SkinThickness,
              "Insulin":Insulin,
              "BMI" :BMI,
              "DiabetesPedigreeFunction" :DiabetesPedigreeFunction,
              "Age" :Age,
              "Pregnancies" :Pregnancies,
              "smoking" :smoking,
              "anaemia" :anaemia,
              "Sex" :Sex}
    
                
    features=pd.DataFrame(user_data,index=[0])
    return features


user_input=get_user_input()

st.subheader("User Input Data:")
st.write(user_input)


log=KNeighborsClassifier()
log.fit(X_train,Y_train)


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




st.subheader("Data Visualization using Pyplot:")
image=Image.open("diavisualization.png")

st.image(image,width=800,cmap="dark12")
st.subheader("Prediction of the Data:")
image1=Image.open("diasex.jpg")
image2=Image.open("diaglucose.jpg")
image3=Image.open("diainsulin.jpg")
image4=Image.open("diabp.jpg")
#st.subheader("Hello")
image5=Image.open("diast.jpg")
#st.subheader("Hry")
image6=Image.open("diabmi.jpg")
image7=Image.open("diadpf.jpg")
image8=Image.open("diaage.jpg")
image9=Image.open("diapre.jpg")
image10=Image.open("diasmoking.jpg")
image11=Image.open("diaanaemia.jpg")
image12=Image.open("diasex.jpg")
st.image([image1,image2,image3,image4,image5,image6,image7,image8,image9,image10,image11,image12],width=300)






st.subheader("Prevalence of Diabetes:")
if st.button("Click Here.."):
    labels = ['Bihar','Manipur','Assam','Jarkhand','Meghalaya','Tripura','Mizoram','Andhara Pradesh','Arunachal Pradesh','Karnataka','Punjab','Gujart','Tamil Nadu','Maharashtra','Chandigarh']
    values = [4.3,5.1,5.5,5.3,4.5,9.4,5.8,8.4,5.1,7.5,10.0,7.1,10.4,8.4,13.6]

    fig = go.Figure(data=[go.Pie(labels=labels,title='PREVALENCE OF DIABETES',values=values)])
    fig.show()



st.subheader("Prevalence of Diabetes in Male and Female")
if st.button("Click Here..."):

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    labels = ["20-25", "25-30", "30-35", "35-40", "40-45", "45-50",
          "50-55","55-60","60-65","65-70","70-75","75-80"]

    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(labels=labels, values=[2,3,4,6,8,10,12,14,16,18,19], name="MALE"),
              1, 1)
    fig.add_trace(go.Pie(labels=labels, values=[2,4,5,7,10,14,16,19,20,21,25], name="FEMALE"),
              1, 2)

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")

    fig.update_layout(
    title_text="DIABETES PREVALENCE (%)",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='MALE', x=0.18, y=0.5, font_size=20, showarrow=False),
                 dict(text='FEMALE', x=0.82, y=0.5, font_size=20, showarrow=False)])   
    fig.show()
