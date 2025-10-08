from django.urls import path
from .views import predict_caption

urlpatterns = [
    path("predict_caption/", predict_caption, name="predict_caption"),
]