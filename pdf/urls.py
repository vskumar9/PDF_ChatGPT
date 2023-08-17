from django.urls import path
from .views import pdf_chat

urlpatterns = [
    path('pdf_chat/', pdf_chat, name='pdf_chat'),
]