import json
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from rest_framework.decorators import api_view
# Create your views here.

from Inference.dnn import InventoryForecasting
from Inference.util import load_Data, get_inference_data

model = InventoryForecasting()
model.run()

Xtest, minmax_scaler = load_Data()[2:]

@csrf_exempt
@api_view(['GET', 'POST'])
def predict(request):
    if request.method == 'POST':
        sample_input = request.data
        x = get_inference_data(Xtest, sample_input)
        p = model.Inference(x.reshape(-1,1))
        return Response({'sales' : p})