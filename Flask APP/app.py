import os
import json
import requests
import numpy as np
import pandas as pd
from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)

from dnn import InventoryForecasting
from util import load_Data, get_inference_data
from variables import port, host

Xtest, minmax_scaler = load_Data()[2:]
model = InventoryForecasting()
model.run()

def make_predictions(sample_input):
    x = get_inference_data(Xtest, sample_input)
    p = model.Inference(x.reshape(1,-1))
    return int(minmax_scaler.inverse_transform(p).squeeze())

@app.route("/predict", methods=['GET','POST'])
def predict():
    message = request.get_json(force=True)
    sales = make_predictions(message)
    response = {
            'sales'   : sales,
            }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host=host, port=port, threaded=False, use_reloader=False)