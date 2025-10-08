from django.urls import path
from .views import predict_image_sentiment

urlpatterns = [
    path("predict_image_sentiment/", predict_image_sentiment),
]
