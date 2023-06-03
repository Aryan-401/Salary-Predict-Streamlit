import pandas as pd
import streamlit as st
import pickle
import numpy as np
import torch
import torch.nn as nn

# Load the trained model

COUNTRIES = ['Australia', 'Brazil', 'Canada', 'France', 'Germany', 'India', 'Italy', 'Netherlands', 'Other', 'Poland',
             'Spain', 'United Kingdom of Great Britain and Northern Ireland', 'United States of America']

EDUCATION = ['Bachelors', 'Less than a Bachelors', 'Masters', 'Post Graduation']


class SalaryPredict(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.l1 = nn.Linear(n_input_features, 128)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(64, 32)
        self.relu3 = nn.LeakyReLU()
        self.l4 = nn.Linear(32, 16)
        self.relu4 = nn.LeakyReLU()
        self.l5 = nn.Linear(16, 8)
        self.relu5 = nn.LeakyReLU()
        self.l6 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        x = self.relu3(x)
        x = self.l4(x)
        x = self.relu4(x)
        x = self.l5(x)
        x = self.relu5(x)
        x = self.l6(x)
        return x


def load_model(pytorch_model=SalaryPredict):
    with open('model_regressor.pkl', 'rb') as file:
        data = pickle.load(file)
    loader = torch.load('model_15k.pt')
    model = pytorch_model(data['n_input'])
    model.load_state_dict(loader['model'])
    data['pytorch'] = model
    return data


def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("""### We need some information to predict the salary""")
    st.divider()

    country = st.selectbox("Country", COUNTRIES)
    education = st.selectbox("Education Level", EDUCATION)
    age = st.slider("Age", 16, 80, 18)
    experience = st.slider("Years of Experience", 0, 50, 1)

    ok = st.button("Calculate Salary")
    if ok:
        param = np.array([[country, age, education, experience]])
        #create a dataframe with column names
        st.table(pd.DataFrame(param,columns=['Country', 'Age', 'Education', 'Experience']))
        param[:, 0] = le_country.transform(param[:, 0])
        param[:, 2] = le_ed.transform(param[:, 2])
        param = param.astype(float)
        st.divider()
        st.subheader("Predicting Salary using Various Models")
        lp, dp, rp, gp, nnp = round(linear.predict(param)[0], 2), round(decision.predict(param)[0], 2), \
            round(randomforest.predict(param)[0], 2), round(regressor.predict(param)[0], 2), round(neuralnet(
            torch.from_numpy(param).float()).item(), 2)
        st.table(pd.DataFrame([lp, dp, rp, gp, nnp], columns=['Salary'], index=['Linear', 'Decision Tree',
                                                                                'Random Forest', 'Grid Search',
                                                                                'Neural Network']))
        st.write(f'Mean Salary from all models: $ {np.mean([lp, dp, rp, gp, nnp]):,.2f}')


data = load_model()
linear = data['Linear']
decision = data['Decision']
randomforest = data['RandomForest']
regressor = data['Grid']
le_country = data['le_country']
le_ed = data['le_ed']
neuralnet = data['pytorch']
