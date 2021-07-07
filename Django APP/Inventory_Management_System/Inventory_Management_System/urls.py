from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('Inference/', include("Inference.urls"), name="predict")
]
