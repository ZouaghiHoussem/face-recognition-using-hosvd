#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 17:08:07 2017

@author: marin
"""

import keras
from keras.datasets import mnist, cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import regularizers
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os
#import imageio as img
import time

#imgShapeOrg = imageio.imread('att_faces/orl_faces/s1/1.pgm').shape
imgShapeOrg = img.imread('att_faces/orl_faces/s1/1.pgm').shape
imgShape = (imgShapeOrg[0], imgShapeOrg[1])

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_face_recogniton_trained_model.json'
weights_name = 'keras_face_recogniton_trained_weights.h5'
results_file = 'results.txt'
data_path = 'att_faces/orl_faces'


P = 40   #Broj osoba
I = 7  #broj slika
Type = 3

batch_size = 10
epochs = 15
data_augmentation = True


#----------------------RETRAIN FUNCTIONS-------------------------------

FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172


def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) 
  predictions = Dense(nb_classes, activation='softmax')(x) 
  model = Model(input=base_model.input, output=predictions)
  return model

def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop',    
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
  
def setup_to_finetune(model):
   """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top 
      layers.
   note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in 
         the inceptionv3 architecture
   Args:
     model: keras model
   """
   for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
      layer.trainable = False
   for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
      layer.trainable = True
   model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),   
                 loss='categorical_crossentropy')


#------------------------------------------------------------------


def LoadData(P, I, imgSize):
    X = np.zeros((P*I, imgSize[0], imgSize[1], 1))
    Y = np.zeros(P*I)
    x_test = np.zeros((P*(10-I), imgSize[0], imgSize[1], 1))
    y_test = np.zeros(P*(10-I))
    i=0; j=0      
    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            p = subdir[-2:]
            if p[0] == 's':
                p = p[-1:]
            tmp = img.imread(os.path.join(subdir,file))
            if int(file[:-4]) > I:
                x_test[j,:,:,0] = tmp
                y_test[j] = int(p)-1
                j += 1
            else:
                X[i,:,:,0] = tmp
                Y[i] = int(p)-1
                i += 1
    return X, Y, x_test, y_test


if Type == 1:
    lab=["0","1","2","3","4","5","6","7","8","9"]
    num_classes = len(lab)
elif Type == 2:
    lab=["airplane" ,"automobile" ,"bird" ,"cat","deer","dog","frog","horse","ship","truck"]
    num_classes = len(lab)
else:
    num_classes = P


start = time.time()

# The data, shuffled and split between train and test sets:
if Type == 1:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train=np.reshape(x_train,(-1,28,28,1))
    x_test=np.reshape(x_test,(-1,28,28,1))
elif Type == 2:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
else:
    x_train, y_train, x_test, y_test = LoadData(P, I, imgShape)



"""
if Type == 1:
    plt.imshow(x_train[150,:,:,0],cmap="gray")
else:
    plt.imshow(x_train[85,:,:,:])
"""

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train=x_train/255
x_test=x_test/255
#x_train -= np.mean(x_train, axis = 0) # zero-center
#x_test -= np.mean(x_test, axis = 0) # zero-center
#x_train /= np.std(x_train, axis = 0) # normalize
#x_test /= np.std(x_test, axis = 0) # normalize
#x_train = np.fft.fft(x_train, norm='ortho') #fourier
#x_test = np.fft.fft(x_test, norm='ortho') #fourier


def initModel():
    model = Sequential()   #inicijaliziramo model
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:])) #konvolucijski sloj (32 filtera dimenzija 3x3
    #-- padding=1). U kerasu padding="same" je isto što i padding=1 dok je padding="valid" znaci padding je 0. Za ostale vrijednosti
    # padding treba koristi neki drugi library (Tensorflow), rijetko se koriste ostale vrijednosti. Za prvi sloj uvijek treba
    # definirat input shape (dimenzije ulaza)
    model.add(Activation('relu')) #Aktivacija
    #model.add(Conv2D(32, (3, 3))) #Kovolucija (32 filtera, 3x3, padding je po deafultu valid)
    #model.add(Activation('relu')) #aktivacija
    model.add(MaxPooling2D(pool_size=(2, 2))) # max-pooling sloj sa prozorom 2x2, stride je automatski namjesten na velicinu prozora
    
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    #model.add(Conv2D(64, (3, 3)))
    #model.add(Activation('relu'))
    #model.add(Conv2D(128, (3, 3)))
    #model.add(Activation('relu'))
    #model.add(Conv2D(128, (3, 3)))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
      
    model.add(Flatten())  #Sliku "spljosti" u vektor da je možemo gurnuti u fully conected layer
    
    model.add(Dense(200, kernel_regularizer=regularizers.l2(0.0), activation='relu'))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    
    model.add(Dense(100, kernel_regularizer=regularizers.l2(0.0), activation='relu'))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    
    #model.add(Dense(8))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    
    #model.add(Dense(80))  # fully connected layser sa 32 neurona
    #model.add(Activation('relu')) # aktivacija
    #model.add(Dropout(0.5))
    
    model.add(Dense(num_classes))  # fully connected sa 10 neurona jer predivdamo 10 kategorija
    model.add(Activation('softmax')) #softmax aktivacija koja brojeve pretvori u vjerojatnosti za pojedinu kategoriju
    return model

def compileModel(model):
    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.001)  #biramo optimizer najcesce se koriste RMS i Adam, learnign_rate postavimo na 0.001
    
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    #Compilamo model, za loss funckiju koristimo 'categorical_crossentropy' koja se skoro pa uvijek koristi kada se predvidaju klase


def train(model):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
              shuffle=True)

  
def loadModel(): 
    # load json and create model
    json_file = open(save_dir+model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(save_dir+weights_name)
    print("Loaded model from disk")
    return loaded_model


def saveModel(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(save_dir+model_name, "w") as json_file:
        json_file.write(model_json)
    
    # serialize weights to HDF5
    model.save_weights(save_dir+weights_name)
    print("Saved model to disk")
   

def saveResults(scores_test, scores_train):
    with open(results_file, 'w') as text_file:
        text_file.write('Test loss: '+ str(scores_test[0]) +'\n')
        text_file.write('Test accuracy: '+ str(scores_test[1]) +'\n')
        text_file.write('Train accuracy: '+ str(scores_train[1]) +'\n')
        text_file.write('err(test) - err(train): '+ str(scores_train[1] - scores_test[1]) +'\n')
        
   

model = initModel()
#model = loadModel()
compileModel(model)
train_start = time.time()
train(model)
train_end = time.time()
#saveModel(model)


if Type == 1 or Type == 2:
    from random import randint
    R = randint(0,x_train.shape[0]-5)
    
    #pogledajmo par predvidanja
    for i in range(R,R+5):
        print("Ulazna slika je ",lab[np.argmax(y_train[i,:])])
        if Type == 1:
            plt.imshow(x_train[i,:,:,0],cmap="gray")
        else:
            plt.imshow(x_train[i,:,:,:])
        plt.show()
        res=model.predict(x_train[i:i+1,:,:,:])
        plt.bar( [1,2,3,4,5,6,7,8,9,10],res[0], align='center')
        plt.xticks([1,2,3,4,5,6,7,8,9,10], lab,rotation='vertical')
        plt.show()
        print("Model nam predviđa: ", lab[np.argmax(res)])
        print("\n")
        #np.argmax uzima index argumenta koji ima najvecu vjerojatnost
        

end = time.time()

print("Vrijeme izvođenja: ", end - start)
print("Vrijeme treniranja: ", train_end - train_start)


# Score trained model.
scores_test = model.evaluate(x_test, y_test, verbose=1)
scores_train = model.evaluate(x_train, y_train, verbose=1)
print('Test loss:', scores_test[0])
print('Test accuracy:', scores_test[1])
print('Train accuracy:', scores_train[1])
print('err(test) - err(train):', scores_train[1] - scores_test[1])
saveResults(scores_test, scores_train)
