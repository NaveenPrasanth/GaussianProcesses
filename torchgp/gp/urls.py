"""torchgp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from . import views
from .service import anon_session
from .service import load_input
urlpatterns = [
    path('', views.index, name='index'),
    path('session/', anon_session.create_session_cookie, name='create_session'),
    path('uploadInput/', load_input.load_input_in_cache, name='input_loader'),
    path('inputPeek/', load_input.peek_input, name='input_peek')

]
