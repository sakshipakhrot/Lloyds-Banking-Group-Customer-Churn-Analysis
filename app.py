import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle



with open('model.pkl','rb') as file:
    model=pickle.load(file)



st.title('Customer Churn Prediction')


Number_of_Interactions = st.number_input('Number of Interactions')
Number_of_Transaction = st.number_input('Number of Transaction')
total_amount_spent = st.number_input('Total Amount Spent')
LoginFrequency = st.number_input('Login Frequency')


main_churn = pd.DataFrame({
    'Number_of_Interactions': [Number_of_Interactions],
    'Number_of_Transaction': [Number_of_Transaction],
    'total_amount_spent': [total_amount_spent],
    'LoginFrequency': [LoginFrequency]
})




prediction = model.predict(main_churn)
prediction_proba = prediction[0]

st.write(f'Churn Probability: {prediction_proba:.2f}')


if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')