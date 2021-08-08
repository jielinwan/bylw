# create data and label for CK+
#  0=anger 1=disgust, 2=fear, 3=happy, 4=sadness, 5=surprise, 6=Neutral
# contain 240,240,240,240,240,240,480  images

import csv
import os
import numpy as np
import h5py
import skimage.io

ck_path = '../data/Oulu/g_crop_144/'

anger_path = os.path.join(ck_path, 'Anger')
disgust_path = os.path.join(ck_path, 'Disgust')
fear_path = os.path.join(ck_path, 'Fear')
happy_path = os.path.join(ck_path, 'Happiness')
sadness_path = os.path.join(ck_path, 'Sadness')
surprise_path = os.path.join(ck_path, 'Surprise')
neutral_path = os.path.join(ck_path, 'Neutral')

# # Creat the list to store the data and label information
data_x = []
data_y = []

datapath = os.path.join('data','g_Oulu_144_data.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

# order the file, so the training set will not contain the test set (don't random)
files = os.listdir(anger_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(anger_path,filename))
    data_x.append(I.tolist())
    data_y.append(0)

files = os.listdir(disgust_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(disgust_path,filename))
    data_x.append(I.tolist())
    data_y.append(1)

files = os.listdir(fear_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(fear_path,filename))
    data_x.append(I.tolist())
    data_y.append(2)

files = os.listdir(happy_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(happy_path,filename))
    data_x.append(I.tolist())
    data_y.append(3)

files = os.listdir(sadness_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(sadness_path,filename))
    data_x.append(I.tolist())
    data_y.append(4)

files = os.listdir(surprise_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(surprise_path,filename))
    data_x.append(I.tolist())
    data_y.append(5)

files = os.listdir(neutral_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(neutral_path,filename))
    data_x.append(I.tolist())
    data_y.append(6)

print(np.shape(data_x))
print(np.shape(data_y))

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("data_pixel", dtype = 'uint8', data=data_x)
datafile.create_dataset("data_label", dtype = 'int64', data=data_y)
datafile.close()

print("Save data finish!!!")
