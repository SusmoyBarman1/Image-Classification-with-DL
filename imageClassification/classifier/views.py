from django.shortcuts import render

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
import os

main_dir = os.path.dirname(os.path.abspath(__file__))

train_dir = "input/train"
path = os.path.join(main_dir,'input')
path = os.path.join(path,'train')


X = []
y = []
convert = lambda category : int(category == 'dog')
def create_test_data(path):
    for p in os.listdir(path):
        category = p.split(".")[0]
        category = convert(category)
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X.append(new_img_array)
        y.append(category)


create_test_data(path)
X = np.array(X).reshape(-1, 80,80,1)
y = np.array(y)

#Normalize data
X = X/255.0

model = Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = X.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2)))
# Add another:
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)


def index(request):
    from .models import User
    users = User.objects.all();
    p = users[len(users)-1].pic
    print(p.url)

    return render(request,'index.html', {'pic':p.url})

def uploadImage(request):
    picture = request.FILES['image'];

    from .models import User

    
    user = User(pic=picture);
    user.save();

    users = User.objects.all();
    p = users[len(users)-1].pic


    temp = p.url
    temp1 = temp.split('/')[-1]
    path = os.path.join(main_dir,'media')
    path = os.path.join(path,'photo')
    #path = os.path.join(path,temp1[-1])

    X_test = []
    id_line = []
    print()
    print('--------------working on------------')
    def create_test1_data(path,temp1):
        
        for p in os.listdir(path):
            if p==temp1:
                print(p)
                print()
                id_line.append(p.split(".")[0])
                img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
                new_img_array = cv2.resize(img_array, dsize=(80, 80))
                X_test.append(new_img_array)
        
    create_test1_data(path,temp1)
    dem = X_test
    X_test = np.array(X_test).reshape(-1,80,80,1)
    X_test = X_test/255.0

    predictions = model.predict(X_test)
    predicted_val = [int(round(p[0])) for p in predictions]

    ans = ""

    print('----------------Uploaded----------------- ')
    print(p.url)
    print()

    if predicted_val[0]==0:
        print()
        print("---------------Cat-------------------")
        print()
        print()
        ans = "Cat"
    elif predicted_val[0]==1:
        print()
        print("-------------------Dog----------------")
        print()
        print()
        ans = "Dog"

    return render(request,'index.html')