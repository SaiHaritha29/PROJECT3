import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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

df=pd.read_csv("Healthcare Datasets 2020.csv")


    #pregnancies=st.sidebar.slider.markdown("<h1 style='color: Blue;'>'pregnancies', 0, 17, 3</h1>",unsafe_allow_html=True)
Cold=st.sidebar.selectbox("Prescribed",('Cold tablet 4mg','Tylenol','Benadryl allergy plus congestion','Ibuprofen(Advil)','Phenylephrine','Crocin cold and flu',
'SBL Nux Vomica',
'SBL Euphrasia Officinalis Dilution 30 CH',
'Sudafed',
'Afrin Spray'))
if Cold=='Cold tablet 4mg':
    st.subheader("Dosage:")
    st.success("Dosage for adults and children 12 years and over Take one tablet every four hours Do not take more than six tablets in 24 hours.")
    st.subheader("Side Effects:")
    st.error("Drowsiness, dizziness, constipation, stomach upset, blurred vision, or dry mouth/nose/throat")
    st.subheader("Patient Review:")
    st.info("72%")
    st.subheader("Price:")
    st.warning("Rs:50")
