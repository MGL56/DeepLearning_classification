#!/usr/bin/env python
# coding: utf-8
# %%
from sklearn.model_selection import KFold, StratifiedKFold
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, MaxPool2D
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import os
import numpy as np


from tensorflow.keras.applications.resnet_v2 import preprocess_input

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL, space_eval
import glob
import time
from keras.optimizers import Nadam, Adam
from ImageAugmented import *


# %%
class MaxConv(Sequential):
    
    def __init__(self, n_classes=5, **kwargs):

        super().__init__()
                
        defaults = {'activation': 'relu','padding': 'same', 'pool_size': (3, 3), 'n_neurons': 4096,
                    'maxpool_strides': (2, 2), 'conv2d_strides': (1, 1), 'n_conv2D': 1, 'kernel_size': (3, 3)}
        defaults = {k:v for k, v in defaults.items() if k not in kwargs}
        params = {**kwargs, **defaults}    
    
        self.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation=params['activation'], padding='same', input_shape=(224, 224, 3)))
        self.add(BatchNormalization())
        self.add(MaxPool2D(pool_size=params['pool_size'], strides=params['maxpool_strides'], padding='valid'))
        
        self.add(Conv2D(filters=256, kernel_size=params['kernel_size'], strides=params['conv2d_strides'], activation=params['activation'], padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D(pool_size=params['pool_size'], strides=params['maxpool_strides']))
        
        
        while params['n_conv2D']>0: 
            self.add(Conv2D(filters=384, kernel_size=params['kernel_size'], strides=params['conv2d_strides'], activation=params['activation'], padding='same'))
            self.add(BatchNormalization())
            params['n_conv2D'] -= 1 

        self.add(Conv2D(filters=256, kernel_size=params['kernel_size'], strides=params['conv2d_strides'], activation=params['activation'], padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D(pool_size=params['pool_size'], strides=params['maxpool_strides'], padding='valid'))
        
        
        self.add(Flatten())
        self.add(Dense(params['n_neurons'], activation=params['activation']))
        self.add(Dropout(0.5))
        self.add(Dense(n_classes, activation='softmax'))


# %%
class MyModel(Sequential):
    
    def __init__(self, model=MaxConv(), optimizer='Adam', callbacks=[], epochs=1, verbose=2):
        
        super().__init__()
        self.epochs = epochs
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.verbose = verbose
        
        for layer in model.layers:
            self.add(layer)
            
        self.compile(loss='sparse_categorical_crossentropy',
                     optimizer=self.optimizer,
                     metrics=['accuracy'])
          
    def fit(self, X=None, y=None, validation_split=0.0, validation_data=None):
        super().fit(X,
                    y,
                    validation_split=validation_split,
                    validation_data=validation_data,
                    epochs=self.epochs,
                    callbacks=self.callbacks,
                    verbose=self.verbose)

    def plot_results(self):
        
        
        # Calcul des moyennes glissantes
        
        step = 20
        while self.n_epochs - step < 0:
            step -= 1
        list_epochs = range(self.n_epochs - step)
            
        loss = [np.mean(self.history.history['loss'][k: k + step]) for k in list_epochs]
        val_loss = [np.mean(self.history.history['val_loss'][k: k + step]) for k in list_epochs]
        
        acc = [np.mean(self.history.history['accuracy'][k: k + 20]) for k in list_epochs]
        val_acc = [np.mean(self.history.history['val_accuracy'][k: k + 20]) for k in list_epochs]
        
        fig, ax = plt.subplots(2, 1, figsize=(6, 6))
        ax[0].plot(loss, label="TrainLoss")
        ax[0].plot(val_loss, label="ValLoss")
        ax[0].legend(loc='best', shadow=True)

        ax[1].plot(acc, label="TrainAcc")
        ax[1].plot(val_acc, label="ValAcc")
        ax[1].legend(loc='best', shadow=True)
        plt.show()

    def create_submit(self, test_data):
        'Create basic file submit'
        #self.model.load_weights("./model_weights.hdf5")
        # predict on data
        results = self.predict_generator(test_data)

        # binarize prediction
        # rbin = np.where(results > 0.5, 1, 0)

        # save results to dataframe
        results_to_save = pd.DataFrame({"id": test_data.images_paths,
                                        "label": results[:,0]
                                        })

        results_to_save["id"] = results_to_save["id"].apply(lambda x: x.replace("../input/test/", "").replace(".tif", ""))

        # create submission file
        results_to_save.to_csv("./submission.csv", index=False)
        return results_to_save

