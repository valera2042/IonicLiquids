# -*- coding: utf-8 -*-
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('lili.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index1.html', prediction_text='Ignition time of your molecule is {} ms'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

