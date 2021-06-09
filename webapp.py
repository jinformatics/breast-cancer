import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math
import os

port = int(os.environ.get('PORT' ,5000))


app = Flask(__name__)
model = pickle.load(open('breast-cancer.pickle','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features  = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    if (prediction[0]==0):
      return render_template('index.html',prediction_text = "patient has cancer (malignent tumor)")
    else:
        return render_template('index.html',prediction_text = "patient has no cancer (malignent benign)")


if __name__ == '__main__':
    app.run(host = '0.0.0.0' , port = port , debug = True)
