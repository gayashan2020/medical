from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib as jb
import pandas as pd
import json
scaler_fileH = "risk/scalerH.save"
scaler_fileP = "risk/scalerP.save"
scaler_fileW = "risk/scalerW.save"

model_fileH = "risk/model_fileH.save"
model_fileP = "risk/model_fileP.save"
model_fileW = "risk/model_fileW.save"

scalerH = jb.load(scaler_fileH)
scalerP = jb.load(scaler_fileP)
scalerW = jb.load(scaler_fileW)

classifierH = jb.load(model_fileH)
classifierP = jb.load(model_fileP)
classifierW = jb.load(model_fileW)

# data = pd.read_csv('risk/Heart_attack.csv')
# dataSource = []
# for i in data.index:
#     record = data.iloc[i].to_json()
#     dataSource.append(record)

# print(dataSource)

def get_risk_level(age,Gender,Cholesterol,Pulse,Smoke,Alcohol,Shortness_of_breath,Anxiety):
    age = age
    Gender = Gender
    Cholesterol = Cholesterol
    Pulse = Pulse
    Smoke = Smoke
    Alcohol = Alcohol
    Shortness_of_breath = Shortness_of_breath
    Anxiety = Anxiety

    valH = [age, Gender, Cholesterol, Pulse, Smoke, Alcohol]
    valP = [age, Gender, Shortness_of_breath, Pulse, Smoke, Alcohol]
    valW = [age, Gender, Anxiety, Shortness_of_breath, Smoke, Alcohol]

    valH = scalerH.transform([valH])
    valP = scalerH.transform([valP])
    valW = scalerH.transform([valW])
    y_predictH = classifierH.predict(valH)
    y_predictP = classifierH.predict(valP)
    y_predictW = classifierH.predict(valW)

    return y_predictH,y_predictP,y_predictW
    # print('Heart Attack :',y_predictH, 'Pneumonia :',y_predictP, 'Wheezing :',y_predictW)

