# may 14th 2026




import os

data_path = r"C:\credit_risk_analysis_project\Data"
print(os.listdir(data_path))

import pandas as pd
df = pd.read_csv(r"C:\credit_risk_analysis_project\Data\german.data",
                sep = r"\s+",
                header = None,
                engine = "python")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

df.columns = [
    "CheckingAccountStatus",
    "DurationMonths",
    "CreditHistory",
    "Purpose",
    "CreditAmount",
    "SavingsAccount",
    "EmploymentSince",
    "InstallmentRate",
    "PersonalStatusSex",
    "OtherDebtors",
    "ResidenceSince",
    "Property",
    "Age",
    "OtherInstallmentPlans",
    "Housing",
    "ExistingCredits",
    "Job",
    "PeopleLiable",
    "Telephone",
    "ForeignWorker",
    "CreditRisk"
]

print(df.head())
print(df["CreditRisk"].value_counts())
print(df.isnull().sum())

#EDA [Actual Eploratory Data Analysis]
print(df["CreditRisk"].value_counts(normalize=True)*100)

print(df["CreditRisk"])

risk_numeric_analysis = df.groupby("CreditRisk")[["DurationMonths", "CreditAmount","Age"]].mean()
print(risk_numeric_analysis )

import matplotlib.pyplot as plt
df["CreditRisk"].value_counts().plot(kind = "bar")
plt.title("Credit Risk Distribution")
plt.xlabel("Credit Risk")
plt.ylabel("Number of Customers")
plt.show()

df.groupby("CreditRisk")["CreditAmount"].mean().plot(kind="bar")
plt.title("Average Credit Amount at Risk")
plt.xlabel("Credit Risk")
plt.ylabel("Average Credit Amount")
plt.show()

df.groupby("CreditRisk")["DurationMonths"].mean().plot(kind="bar")
plt.title("Avg Duration Loan by Risk")
plt.xlabel("Credit Risk")
plt.ylabel("Avg Duration(Months)")
plt.show()

df.groupby("CreditRisk")["Age"].mean().plot(kind="bar")
plt.title("Avg age by Credit Risk")
plt.xlabel("Credit Risk")
plt.ylabel("Avg Age")
plt.show()

df.groupby("CreditRisk")["InstallmentRate"].mean().plot(kind = "bar")
plt.title("Avg Installmentrate by Credit Risk")
plt.xlabel("Credit Risk")
plt.ylabel("Avg Installment rate")
plt.show()


housing_risk = df.groupby(['Housing',"CreditRisk"]).size()
print(housing_risk)

def categorical_risk_analysis(column_name):
    print("\n" + "="*50)
    print(column_name, "vs Credit Risk")
    print("="*50)

    risk_count = df.groupby([column_name, "CreditRisk"]).size()
    print(risk_count)

    risk_count.unstack().plot(kind="bar")
    plt.title(column_name + " vs Credit Risk")
    plt.xlabel(column_name)
    plt.ylabel("Number of Customers")
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()


categorical_risk_analysis("CreditHistory")
categorical_risk_analysis("CheckingAccountStatus")
categorical_risk_analysis("SavingsAccount")
categorical_risk_analysis("EmploymentSince")
categorical_risk_analysis("Purpose")

#Feature Engineering
#Early Warning System

df["HighCreditAmountFlag"] = df["CreditAmount"].apply( lambda x:1 if x > df["CreditAmount"].median() else 0)
df["LongDurationFlag"] = df["DurationMonths"].apply( lambda x: 1 if x> df["DurationMonths"].median() else 0)
df["YoungApplicantFlag"] = df["Age"].apply(lambda x :1 if x < 30 else 0)

print(df[["CreditAmount", "HighCreditAmountFlag",
          "DurationMonths", "LongDurationFlag",
          "Age","YoungApplicantFlag"]].head())


#Combined Risk Warning Score
df["RiskWarningScore"] = (
    df["HighCreditAmountFlag"] +
    df["LongDurationFlag"] +
    df["YoungApplicantFlag"]
)

print(df[[
    "HighCreditAmountFlag",
    "LongDurationFlag",
    "YoungApplicantFlag",
    "RiskWarningScore"
]].head())

warning_risk_analysis = df.groupby("RiskWarningScore")["CreditRisk"].value_counts()
print(warning_risk_analysis)

def assign_risk_level(score):
    if score == 0:
        return "Low Risk"
    elif score == 1:
        return "Moderate Risk"
    elif score == 2:
        return "High Risk"
    else:
        return "Critical Risk"

df["RiskLevel"] = df["RiskWarningScore"].apply(assign_risk_level)

print(df[["RiskWarningScore", "RiskLevel", "CreditRisk"]].head())


risk_level_summary = df["RiskLevel"].value_counts()
print(risk_level_summary)

df["RiskLevel"].value_counts().plot(kind="bar")

plt.title("Customer Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Customers")

plt.xticks(rotation=45)

plt.show()


#encoding categorical variables into numbers

from sklearn.preprocessing import LabelEncoder

#Create encoder object
label_encoder = LabelEncoder()

#convert categorical columns into numeric values
for column in df.select_dtypes(include="object").columns:
    df[column] = label_encoder.fit_transform(df[column])

print(df.head())


#FEATURe/TARGET SPLIT
X = df.drop(columns=["CreditRisk", "RiskLevel"])
y = df["CreditRisk"]

print(X.shape)
print(y.shape)

#TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,
    test_size=0.2,
    random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
# Create model
model = LogisticRegression(max_iter=1000)
# Train model
model.fit(X_train, y_train)
print("Model Training Completed")

#MODEL PREDICTIONS
y_pred = model.predict(X_test)
print(y_pred[:10])
print(y_test[:10].values)


#MODEL ACCURACY
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#RANDOM FOREST MODEL

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

#Train model
rf_model.fit(X_train, y_train)

#Predictions
rf_y_pred = rf_model.predict(X_test)

#Accuracy
rf_accuracy = accuracy_score(y_test, rf_y_pred)

print("Random Forest Accuracy:", rf_accuracy)

#Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_y_pred))

#Classification Report
print("\nClassification Report:")
print(classification_report(y_test, rf_y_pred))

#we notice random Forest improved overall accuracy, while Logistic Regression performed slightly better in detecting risky customers.

#Balances Random Forest Model
balanced_rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced")

balanced_rf_model.fit(X_train, y_train)

balanced_rf_pred = balanced_rf_model.predict(X_test)

balanced_rf_accuracy = accuracy_score(y_test, balanced_rf_pred)

print("Balanced Random Forest Accuracy:", balanced_rf_accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, balanced_rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, balanced_rf_pred))

#so we try balanced logisric regression now as for the data we can notice logistic is cirrently better than random forest
# BALANCED LOGISTIC REGRESSION MODEL
balanced_log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced")

balanced_log_model.fit(X_train, y_train)
balanced_log_pred = balanced_log_model.predict(X_test)
balanced_log_accuracy = accuracy_score(y_test, balanced_log_pred)

print("Balanced Logistic Regression Accuracy:", balanced_log_accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, balanced_log_pred))
print("\nClassification Report:")
print(classification_report(y_test, balanced_log_pred))


df["PredictedCreditRisk"] = balanced_log_model.predict(X)
print(df[["CreditRisk","PredictedCreditRisk","RiskLevel"]].head())

df.to_csv(r"C:\credit_risk_analysis_project\data\final_credit_risk_dataset.csv",index=False)
print("Final dataset exported successfully.")