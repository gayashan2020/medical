from pathlib import Path
import cv2
from PIL import Image
from flask import Flask, request, jsonify
from utility.pill_to_cv2 import imgread
app = Flask(__name__)
from model_process import model_process_img
from risk import get_risk_level
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tensorflow.keras.utils import to_categorical
import json

@app.route('/lung', methods = ['GET', 'POST'])
def predict():
    data = {}

    filestr = request.files['file'].read()
    img = imgread(filestr)

    prediction_inst = []
    list_Of_cf = []

    outputs = model_process_img(img)
    for item in outputs['predictions']:
        list_Of_cf.append(item['confidence'])

    for item in outputs['predictions']:
        if item['confidence'] == max(list_Of_cf):
            print(item['label'], max(list_Of_cf) * 100)
            prediction_inst.append(item['label'])

    temp_val = prediction_inst[0]
    print(temp_val)

    prediction_inst.clear()
    list_Of_cf.clear()
    data['detection'] = temp_val

    return jsonify(data)

@app.route('/risk', methods = ['GET', 'POST'])
def predictR():
    data = {}
    post_data = request.json

    age = str(post_data['age'])
    Gender = str(post_data['Gender'])
    Cholesterol = str(post_data['Cholesterol'])
    Pulse = str(post_data['Pulse'])
    Smoke = str(post_data['Smoke'])
    Alcohol = str(post_data['Alcohol'])
    Shortness_of_breath = str(post_data['Shortness_of_breath'])
    Anxiety = str(post_data['Anxiety'])

    y_predictH,y_predictP,y_predictW = get_risk_level(age,Gender,Cholesterol,Pulse,Smoke,Alcohol,Shortness_of_breath,Anxiety)

    data['prediction_heart'] = y_predictH[0]
    data['prediction_wheeze'] = y_predictP[0]
    data['prediction_pneumonia'] = y_predictW[0]

  

    return jsonify(data)

@app.route('/dataH', methods = ['GET', 'POST'])
def predictH():
    data = {}
    data_Heart_attack = pd.read_csv('risk/Heart_attack.csv')
    data_Heart_attack = data_Heart_attack.to_json(orient = "records")

    data['details']=data_Heart_attack

    return jsonify(data)

@app.route('/dataW', methods = ['GET', 'POST'])
def predictW():
    data = {}
    data_Wheezing = pd.read_csv('risk/Wheezing.csv')
    data_Wheezing = data_Wheezing.to_json(orient = "records")

    data['details']=data_Wheezing

    return jsonify(data)

@app.route('/dataP', methods = ['GET', 'POST'])
def predictP():
    data = {}
    data_Pneumonia = pd.read_csv('risk/Pneumonia.csv')
    data_Pneumonia = data_Pneumonia.to_json(orient = "records")

    data['details']=data_Pneumonia

    return jsonify(data)


if __name__ == '__main__':
    print('Loading model...')
    # predict()
    app.run(host='127.0.0.1', port=8081,debug=False)