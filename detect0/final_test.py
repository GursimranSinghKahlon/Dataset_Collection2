################################3
# TRAINING OUR MODEL

# import the necessary packages
import h5py
import numpy as np
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib

# from .global_new import fd_hu_moments
# from .global_new import fd_haralick
# from .global_new import fd_histogram
from pprint import pprint
from time import sleep
import xml.dom.minidom as xml
import shutil

#import urllib3
from urllib.request import urlopen
import re
import os
import random


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import mahotas


def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

    
# fixed-sizes for image
fixed_size = tuple((500, 500))
train_path = os.path.join(os.getcwd(),"detect0","dataset","train")
test_path = os.path.join(os.getcwd(),"detect0","dataset","test")
static_path = os.path.join(os.getcwd(),"detect0","static")


num_trees = 100
# bins for histogram
bins = 8
# train_test_split size
test_size = 0.10
# seed for reproducing same results
seed = 9

# empty lists to hold feature vectors and labels
global_features = []
labels = []

# get the training labels
train_labels = os.listdir(train_path)
train_labels.sort()
#print(train_labels)

file_name3 = ""
def get_image():
    print("download starting")
    for files in os.listdir(test_path):
        os.unlink(test_path + files)
    for files in os.listdir("static"):
        if(files.endswith(".jpg")):
            os.unlink("static/" + files)
    img_data=urlopen('https://firebasestorage.googleapis.com/v0/b/picfi-79b51.appspot.com/o/image.jpg?alt=media&token=###_token_here_###').read()
    rr = random.randint(1,9999)
    filename = test_path + str(rr) + ".jpg"
    file_name3 = str(rr) + ".jpg"
    filename2= "static/" + str(rr) + ".jpg"
    with open(filename, 'wb') as f:
        f.write(img_data)
    with open(filename2, 'wb') as f2:
        f2.write(img_data)
        print("download completed")
        return("OK! downloaded ")
   


def perdict():
    #get_image()
    #train_labels = os.listdir(train_path)
    
    # import the feature vector and trained labels
    h5f_data_path = os.path.join(os.getcwd(),"detect0","output","data.h5")
    h5f_label_path = os.path.join(os.getcwd(),"detect0","output","labels.h5")
    h5f_data = h5py.File(h5f_data_path, 'r')
    h5f_label = h5py.File(h5f_label_path, 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()

    # verify the shape of the feature vector and labels
    print("[STATUS] features shape: {}".format(global_features.shape))
    print("[STATUS] labels shape: {}".format(global_labels.shape))

  #  print("[STATUS] training started...")

    # split the training and testing data
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                              np.array(global_labels),
                                                                                              test_size=test_size,
                                                                                              random_state=seed)


    # filter all the warnings
    import warnings
    warnings.filterwarnings('ignore')


    ####################
    # TESTING OUR MODEL

    import matplotlib.pyplot as plt

    # create the model - Random Forests
    clf  = RandomForestClassifier(n_estimators=100, random_state=9)

    # fit the training data to the model
    clf.fit(trainDataGlobal, trainLabelsGlobal)

    # path to test data
    #test_path = "dataset/test"

    # loop through the test images
    for file in glob.glob(test_path + "/*.jpg"):
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

        # Global Feature extraction
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        # Concatenate global features
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # predict label of test image
        prediction = clf.predict(global_feature.reshape(1,-1))[0]
        print(train_labels[prediction])

        #directory_src = "dataset/test"
        directory_dest = train_path
        prediction_folder = train_labels[prediction]            
        count = 100 
        for files in os.listdir(directory_dest + "/" + prediction_folder):
            count+=1
        
        #source = directory_src+"/"+filename
        source = file
        destination = directory_dest + "/" + prediction_folder + "/new" + str(count) + ".jpg"
        shutil.copyfile(source, destination)
        


        ####################### 
        # Find related images
    
        part_done = [0,0,0,0]
        
        leaf_path = "null"
        flower_path = "null"
        entire_path = "null"
        stem_path = "null"

        for file_related in glob.glob(train_path + "/" + train_labels[prediction] + "/*.xml" ):
            doc = xml.parse(file_related)
            image_part = doc.getElementsByTagName("Content")[0].firstChild.nodeValue
            #print(image_part)

            if(image_part == "Leaf" and part_done[0]==0 ):
                part_done[0] = 1
                file_related = file_related.split(".")[0] +".jpg"
                print("related file : " + file_related)
                leaf_path = file_related

            elif(image_part == "Flower" and part_done[1]==0 ):
                part_done[1] = 1
                file_related = file_related.split(".")[0] +".jpg"
                print("related file : " + file_related)
                flower_path = file_related

            elif(image_part == "Entire"  and part_done[2]==0 ):
                part_done[2] = 1
                file_related = file_related.split(".")[0] +".jpg"
                print("related file : " + file_related)
                entire_path = file_related

            elif(image_part == "Stem" and part_done[3]==0 ):
                part_done[3] = 1
                file_related = file_related.split(".")[0] +".jpg"
                print("related file : " + file_related)
                stem_path = file_related


            for i in part_done:
                if(i==0):
                    break
            else:
                break

        all_links = []
        all_links.append(train_labels[prediction])
        all_links.append(leaf_path)
        all_links.append(flower_path)
        all_links.append(entire_path)
        all_links.append(stem_path)   


        directory_dest = static_path

        source = leaf_path
        destination = directory_dest + "/" + source.split('/')[-1]

        if(source != "null"):
            shutil.copyfile(source, destination)

        source = flower_path
        destination = directory_dest + "/" + source.split('/')[-1]
        if(source != "null"):
            shutil.copyfile(source, destination)

        source = entire_path
        destination = directory_dest + "/" + source.split('/')[-1]
        if(source != "null"):
            shutil.copyfile(source, destination)

        source = stem_path
        destination = directory_dest + "/" + source.split('/')[-1]
        if(source != "null"):
            shutil.copyfile(source, destination)
                                  
        return(all_links)


