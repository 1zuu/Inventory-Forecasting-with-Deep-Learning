import os, io
from PIL import Image
from flask import Flask
from flask import jsonify
from flask import request
from base64 import encodebytes

app = Flask(__name__)

from dnn import InventoryForecasting
from util import load_Data, get_inference_data
from variables import *

Xtest, minmax_scaler = load_Data()[2:]
model = InventoryForecasting()
model.run()

def make_predictions(sample_input):
    x = get_inference_data(Xtest, sample_input)
    p = model.Inference(x.reshape(1,-1))
    return int(minmax_scaler.inverse_transform(p).squeeze())

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r')
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    return encoded_img

@app.route("/predict", methods=['GET','POST'])
def predict():
    message = request.get_json(force=True)
    sales = make_predictions(message)
    response = {
            'sales'   : sales,
            }

    return jsonify(response)

@app.route('/visualize/<plot>', methods=['GET'])
def visualize(plot):
    try:
        if plot == "results":
            encoded_img = get_response_image(model_results_img)
        elif plot == "error":
            encoded_img = get_response_image(error_analysis_img)
        elif plot == "model":
            encoded_img = get_response_image(model_results_img)
        else:
            raise Exception("Invalid plot")

        response = {"encoded image": encoded_img}
        return jsonify(response)

    except:
        return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(
            debug=True, 
            host=host, 
            port=port, 
            threaded=False, 
            use_reloader=True
            )