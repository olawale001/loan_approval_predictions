import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('loan_prediction.csv')
print(df.head())
print(df.columns)
print(df.describe())

df.fillna({'Gender': 'Gender'}, inplace=True)
df.fillna({'Married': 'Married'}, inplace=True)
df.fillna({'Dependent': 'Dependent'}, inplace=True)
df.fillna({'Self_Employed': 'Self_Employed'}, inplace=True)

df.fillna({'LoanAmount': 'LoanAmount'}, inplace=True)
df.fillna({'Loan_Amount_Term': 'Loan_Amount_Term'}, inplace=True)
df.fillna({'Credit_History': 'Credit_History'}, inplace=True)

loan_status_count = df['Loan_Status'].value_counts()
fig_loan_status = px.pie(
    loan_status_count,
    names=loan_status_count.index,
    title='Loan Approval Status'
)
fig_loan_status.show()

gender_status = df['Gender'].value_counts()
fig_gender = px.bar(
    gender_status,
    x=gender_status.index,
    y=gender_status.values,
    title='Gender Distribution'
)
fig_gender.show()