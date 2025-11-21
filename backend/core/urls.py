from django.urls import path
from .views import ChatForwardView

urlpatterns = [
    path("chat/", ChatForwardView.as_view(), name="chat_forward"),
]