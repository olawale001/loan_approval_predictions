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

married_counts = df['Married'].value_counts()
fig_married = px.bar(
    married_counts,
    x=married_counts.index,
    y=married_counts.values,
    title='Maritial Count Distribution'
)
fig_married.show()

education_counts = df['Education'].value_counts()
fig_education = px.bar(
    education_counts,
    education_counts.index,
    y=education_counts.values,
    title='Education Distribution'
)
fig_education.show()

self_employed_count = df['Self_Employed'].value_counts()
fig_self_employed = px.bar(
    self_employed_count,
    x=self_employed_count.index,
    y=self_employed_count.values,
    title='Self Employed Distribution'
)
fig_self_employed.show()

fig_applicant_income = px.histogram(
    df,
    x='ApplicantIncome',
    title='Applicant Income Distribution'
)
fig_applicant_income.show()

fig_income = px.box(
    df,
    x='Loan_Status',
    y='ApplicantIncome',
    color='Loan_Status',
    title='Loan Status vs. ApplicantIncome'
)
fig_income.show()

Q1 = df['ApplicantIncome'].quantile(0.25)
Q3 = df['ApplicantIncome'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df[(df['ApplicantIncome'] >= lower_bound) & (df['ApplicantIncome'] <= upper_bound)]

fig_coapplicant_income = px.box(
    df,
    x='Loan_Status',
    y='CoapplicantIncome',
    color='Loan_Status',
    title='Loan Status vs. CoapplicantIncome'
)
fig_applicant_income.show()