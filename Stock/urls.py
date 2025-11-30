"""
URL configuration for Pradeep project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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
from django.contrib import admin
from django.urls import path
from Basics.views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("about/", about, name="about"),
    #path("new/", new, name="new"),
    path("", home, name="home"),  # Root URL
    path("calcy/", calcy ,name="calcy"),
    path("login/", login ,name="login"),
    path("signup/", signup ,name="signup"),
    #path('verify-email/', verify_email, name='verify_email'),
    #path('resend-otp/', resend_otp, name='resend_otp'),
    path('predict/', predict, name='predict')
    
    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)