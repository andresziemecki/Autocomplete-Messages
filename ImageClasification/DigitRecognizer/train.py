from __future__ import print_function

import os
import argparse
import numpy as np
import tensorflow as tf
import seaborn as sns

from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from lib.data_preparation import LoadData
from lib.io import createFolder

# Initialization the Configuration

#---- Creamos los argumentos que va a recibir desde afuera
parser = argparse.ArgumentParser()

# --- Segmentation configuration
parser.add_argument('-ptr', '--path_train', type=str, default='/home/andres/Documents/DataScience/DataSets/DigitRecognizer/train.csv', help='Path where the dataset train is')
parser.add_argument('-pte', '--path_test', type=str, default='/home/andres/Documents/DataScience/DataSets/DigitRecognizer/test.csv', help='Path where the dataset test is')
parser.add_argument("-st", "--saveto", type=str, default='', help="Save result to a folder. Default = ./results")
parser.add_argument('-nf', '--nfilters', type=int, default=64, help='Nunmber of filters (default: 64)')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size (default: 32)')
parser.add_argument('-e', '--epochs', type=int, default=100, help='Epochs (default: 100)')
parser.add_argument("-g", "--gpus", type=int, default=1, help="# of GPUs to use for training (default = 1)")

# --- Reading the argument configuration
kwargs = vars(parser.parse_args())
print(kwargs)

# --- Parse the segmentation configuration
path_image_train = kwargs['path_train']
path_image_test = kwargs['path_test']
path_save = kwargs['saveto']
nf = kwargs['nfilters']
batch_size = kwargs['batch_size']
epochs = kwargs['epochs']
gpus = kwargs['gpus']

# Create the folder where we are going to save the results
path_save = createFolder(path_save, kwargs)

# LOAD DATA
X_train, X_val, Y_train, Y_val, test = LoadData(path_image_train, path_image_test, 2)

# With data augmentation to prevent overfitting (accuracy 0.99286)
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

# MODEL PREPARATION

print("\nLoad the model...\n")
from lib.Architectures.half_unet import digit_recognizer_model 
model = digit_recognizer_model()

print("Compiing the model...\n")

# Compile the model with multigpu or single gpu
if (gpus > 1):
    with tf.device('/cpu:0'):
        model = multi_gpu_model(model, gpus=gpus)
optimizer = Adam()
model.compile(optimizer=optimizer,
        loss = "categorical_crossentropy",
        metrics=["accuracy"])

print("\nSummary of the model:\n")
model.summary()

print("\nSaving architecture of the model to a png file...\n")

# Save the model as png image
from keras.utils import plot_model
plot_model(model, to_file= path_save + '/architecture_graph.png', show_shapes = True)


print("\nInitializatin the training...\n")

# Train model on dataset
# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size)


print("\nFinalization of the training model and saving the model trained and the history... \n")

model.save(path_save + '/model.h5')
np.save(path_save + '/train_history.npy', history.history)

print("\nFINISHED\n")
