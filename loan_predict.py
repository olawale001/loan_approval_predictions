import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC



df = pd.read_csv('loan_prediction.csv')
print(df.head())
print(df.columns)
print(df.describe())


df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


loan_status_count = df['Loan_Status'].value_counts()
fig_loan_status = px.pie(
    loan_status_count,
    names=loan_status_count.index,
    title='Loan Approval Status'
)
fig_loan_status.show()


fig_gender = px.bar(
    x=df['Gender'].value_counts().index,
    y=df['Gender'].value_counts().values,
    title='Gender Distribution',
    labels={'x': 'Gender', 'y': 'Count'}
)
fig_gender.show()

fig_married = px.bar(
    x=df['Married'].value_counts().index,
    y=df['Married'].value_counts().values,
    title='Marital Status Distribution',
    labels={'x': 'Marital Status', 'y': 'Count'}
)
fig_married.show()


fig_education = px.bar(
    x=df['Education'].value_counts().index,
    y=df['Education'].value_counts().values,
    title='Education Distribution',
    labels={'x': 'Education', 'y': 'Count'}
)
fig_education.show()

fig_income_hist = px.histogram(
    df,
    x='ApplicantIncome',
    title='Applicant Income Distribution',
    labels={'x': 'Applicant Income'}
)
fig_income_hist.show()


fig_income_box = px.box(
    df,
    x='Loan_Status',
    y='ApplicantIncome',
    color='Loan_Status',
    title='Loan_Status vs ApplicantIncome',
    labels={'x': 'Loan Status', 'y': 'Applicant Income'}
)
fig_income_box

Q1 = df['ApplicantIncome'].quantile(0.25)
Q3 = df['ApplicantIncome'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['ApplicantIncome'] >= lower_bound) & (df['ApplicantIncome'] <= upper_bound)]


Q1 = df['CoapplicantIncome'].quantile(0.25)
Q3 = df['CoapplicantIncome'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['CoapplicantIncome'] >= lower_bound) & (df['CoapplicantIncome'] <= upper_bound)]


cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
df = pd.get_dummies(df, columns=cat_cols)



X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



scaler = StandardScaler()
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
print(X_train.head())
print(X_test.head())


model = SVC(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

X_test_df = pd.DataFrame(X_test, columns=X_test.columns)
X_test_df['Loan_Status_Predicted'] = y_pred
print(X_test_df.head())