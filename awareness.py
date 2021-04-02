import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
import seaborn as sns
from matplotlib import pyplot
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




checkbox=st.sidebar.checkbox("active_cases")

if checkbox:
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



