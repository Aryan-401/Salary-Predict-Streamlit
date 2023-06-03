import streamlit as st
from predict import show_predict_page
from explore import show_explore_page


page = st.sidebar.selectbox("Explore the Code or Predict Salary", ["Predicting Salary", "Process Behind Cleaning Data"])

if page == "Process Behind Cleaning Data":
    show_explore_page()
else:
    show_predict_page()
