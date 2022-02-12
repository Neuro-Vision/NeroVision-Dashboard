
# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from .forms import InputForm, LoginForm, SignUpForm
from .utils import *
import nibabel as nib
from django.contrib.auth.decorators import login_required

import numpy as np
from .plot3d import *
from django.urls import reverse
from django.http import HttpResponse, JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import io
import urllib
import base64
import cv2
from django.http import HttpResponseRedirect
import pickle
from .utils import handle_uploaded_file

import mimetypes
import os
from django.http.response import HttpResponse

from django.shortcuts import render
from plotly.offline import plot
from plotly.graph_objs import Scatter
import pathlib
from django.http import FileResponse

from .unet_v2 import *
from django.views.decorators.csrf import csrf_exempt
from sklearn.preprocessing import MinMaxScaler

import itk                                                               
import itkwidgets
from ipywidgets import interact, interactive, IntSlider, ToggleButtons

from io import StringIO
import io
import urllib, base64
from io import BytesIO


def login_view(request):
    form = LoginForm(request.POST or None)

    msg = None

    if request.method == "POST":

        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("/")
            else:
                msg = 'Invalid credentials'
        else:
            msg = 'Error validating the form'

    return render(request, "accounts/login.html", {"form": form, "msg": msg})


def register_user(request):
    msg = None
    success = False

    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get("username")
            raw_password = form.cleaned_data.get("password1")
            user = authenticate(username=username, password=raw_password)

            msg = 'User created - please <a href="/login">login</a>.'
            success = True

            # return redirect("/login/")

        else:
            msg = 'Form is not valid'
    else:
        form = SignUpForm()

    return render(request, "accounts/register.html", {"form": form, "msg": msg, "success": success})


# def dashboard(request) :
#     return render(request, "home/home.html")
@login_required(login_url="/login/")
def dashboard(request):
    filenames = []
    msg = None # to display any message if wrong input or any
    dummy = [] 

    platform = "local"

    if request.method == "POST":
        #If form reques method is post then load the form with post and file requests
        form = InputForm(request.POST,request.FILES)

        if form.is_valid():
            # iterate over FILES object to get all filename
            print(request.FILES)
            if platform == "local" :
                for filename in request.FILES:
                    print(filename)
                    f = handle_uploaded_file(request.FILES[filename])
                    nii_file = nib.load("apps/static/upload/"+f)
                    dummy.append(nii_file.get_fdata()) # it gets data from loaded nii file
                    filenames.append(f)    

                print(filenames)
                filenames = ["flair.nii.gz", "t1.nii.gz", "t1ce.nii.gz", "t2.nii.gz", "predicted.nii"]

                # unet = UNetV2()

                # prediction = unet.predict(filenames)['Prediction'][0]
                # print(type(prediction))
                # # print(prediction)
                # prediction = (prediction).squeeze().cpu().detach().numpy()
                # prediction = np.moveaxis(prediction, (0, 1, 2, 3), (0, 3, 2, 1))
                # wt, tc, et = prediction
                # print(wt.shape, tc.shape, et.shape)
                # prediction = (wt + tc + et)
                # prediction = np.clip(prediction, 0, 1)
                # print(prediction.shape)
                # print(np.unique(prediction))
                # og = nib.load('segmentation/static/upload/flair.nii')
                # nft_img = nib.Nifti1Image(prediction, og.affine)
                # nib.save(nft_img, 'Dashboard/apps/static/upload/predicted'  + '.nii.gz')

                # reader = ImageReader('./data', img_size=128, normalize=True, single_class=False)
                # viewer = ImageViewer3d(reader, mri_downsample=20)
                # fig = viewer.get_3d_scan(0, 't1')
                request.session['filenames'] =  filenames

                #creating animation code
                reader = ImageReader('./data', img_size=128, normalize=True, single_class=False)
                viewer = ImageViewer3d(reader, mri_downsample=20)
                fig = viewer.get_3d_scan(0, filenames)
                request.session['animation'] =  fig.to_html()

                # creating GIF code
                # predicted_data = nib.load(f"apps/static/predicted_files/{filenames[-1]}").get_data()
                # data_to_3dgif = Image3dToGIF3d(img_dim = (120, 120, 78), binary=True, normalizing=False)
                # transformed_data = data_to_3dgif.get_transformed_data(predicted_data)
                # data_to_3dgif.plot_cube(
                #     transformed_data,
                #     title="Title",
                #     make_gif=True,{fig}} 
                #     path_to_save="apps/static/gif_3d/3d.gif"
                # )

                request.session['gif'] = "3d.gif"

                return redirect('/options/')
    else :
        form = InputForm()
    
    return render(request, "home/home.html", {"form": form, "msg": msg})

@login_required(login_url="/login/")
def options(request) :
    if request.method == 'POST':
        option = request.POST['choice']
        if option == 'animation' :
            return redirect('/plot3D/')
        elif option == 'gif' :
            return redirect('/gif3d/')
        elif option == '2d-slice' :
            return redirect('/2d_view/')
        elif option == 'location' :
            return redirect('/tumor_location/')
    else : 
        return render(request, "home/options.html")

@login_required(login_url="/login/")
def plot_3D(request):

    fig = request.session["animation"]

    return render(request, "home/animation.html", context={'fig': fig})

@login_required(login_url="/login/")
def view_gif_3D(request) :
    gif_file_name = request.session['gif']
    return render(request, "home/view_gif.html", context={"gif" : gif_file_name})