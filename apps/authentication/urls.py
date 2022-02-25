# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('', views.dashboard, name="home"),
    path('login/', views.login_view, name="login"),
    path('register/', views.register_user, name="register"),
    path("signout/", LogoutView.as_view(), name="logout"), 
    path("dashboard/", views.dashboard, name="dashboard"),
    path("options/", views.options, name="options"),
    path("plot3D/", views.plot_3D, name="plot3D"),
    path("gif3d/", views.view_gif_3D, name = "gif3d"),
    path("2D_Visualization/", views.twoD_view, name = "2D_Visualization"),
    path("report/", views.tumor_location, name = "report"),
    path('download_animation/', views.download_animation, name='download_animation'),
    path('download_gif/', views.download_gif, name='download_gif'),
    path('download_2d/', views.download_2d_plots, name='fetch_2d_plots'),
    path('logout/', views.delete_files, name='delete_files'),
]
