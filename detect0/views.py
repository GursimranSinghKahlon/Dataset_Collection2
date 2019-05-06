from django.shortcuts import render
from django.http import HttpResponse
from .models import Product

# Create your views here.
import base64
from PIL import Image
from io import BytesIO
import json
import codecs
from . import final_test
#from description import description
import numpy as np
import re
import os
import shutil


static_path = os.path.join(os.getcwd(),"detect0","static")
static_path2 = "../static"
test_path = os.path.join(os.getcwd(),"detect0","dataset","test")
train_path = os.path.join(os.getcwd(),"detect0","dataset","train")


print(static_path)
print(test_path)

def index(request):
    products = Product.objects.all()
    return render(request, 'index.html',
    {'products' : products})

def update_db(request):
    train_labels = os.listdir(train_path)
    print(train_labels)

    for training_name in train_labels:
        direct = os.path.join(train_path, training_name)
        count = 0
        for filename in os.listdir(direct):
            if filename.endswith(".jpg"):
                count+=1
        
        Product.objects.update_or_create(name=training_name,stock=count)

    products = Product.objects.all()
    return render(request, 'index.html',
    {'products' : products})

def new(request):

    all_links2 = final_test.perdict()
    category = all_links2[0]
    leaf_path2 = os.path.join(static_path2 , all_links2[1].split('/')[-1])
    flower_path2 = os.path.join(static_path2 , all_links2[2].split('/')[-1])
    entire_path2 = os.path.join(static_path2 , all_links2[3].split('/')[-1])
    stem_path2 = os.path.join(static_path2 , all_links2[4].split('/')[-1])

    test_file = ""
    for files in os.listdir(test_path):
        test_file = files	
        shutil.copyfile(test_path + "/" +files,static_path +"/"+files)

    name = category
    plant_name = name
        

    plant_img = name
    res = {}
    res["category"] = "File uploaded belong to :  " + name

    res["leaf_path2"] = leaf_path2
    res["flower_path2"] = flower_path2
    res["entire_path2"] = entire_path2
    res["stem_path2"] = stem_path2
    test_file = "../static/" + test_file
    res["test_file"] = test_file
    res["plant_name"] = plant_name

    print(res)

    return render(request, 'index2.html',res)


def new3(request):
    return HttpResponse("New")

