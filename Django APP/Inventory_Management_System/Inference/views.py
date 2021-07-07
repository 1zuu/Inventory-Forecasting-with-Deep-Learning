import json
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from rest_framework.decorators import api_view

from Inference.dnn import InventoryForecasting
from Inference.util import load_Data, get_inference_data

Xtest, minmax_scaler = load_Data()[2:]
model = InventoryForecasting()
model.run()

def make_predictions(sample_input):
    x = get_inference_data(Xtest, sample_input)
    p = model.Inference(x.reshape(1,-1))
    return int(minmax_scaler.inverse_transform(p).squeeze())

@csrf_exempt
@api_view(['GET', 'POST'])
def predict(request):
    if request.method == 'POST':
        sample_input = request.data
        output = make_predictions(sample_input)
        return Response({'sales' : output})