from __future__ import print_function

import os
import sys
import keras
import random
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



from keras import backend as K
from lib.io import *
from lib.Architectures import *

#---- Creamos los argumentos que va a recibir desde afuera
parser = argparse.ArgumentParser()

# --- Segmentation configuration

parser.add_argument('-p', '--path', type=str, default='', help='Path where the dataset is')
parser.add_argument('-nf', '--nfilters', type=int, default=64, help='Nunmber of filters (default: 64)')
parser.add_argument('-bs', '--batch_size', type=int, default=72, help='Batch size (default: 32)')
parser.add_argument('-e', '--epochs', type=int, default=100, help='Epochs (default: 100)')
parser.add_argument("-g", "--gpus", type=int, default=1, help="# of GPUs to use for training (default = 1)")
parser.add_argument("-is", "--input_shape", type=str, default='64x64', help="Input shape from de Dataset")
parser.add_argument("-sh", "--shuffle", type=bool, default=True, help="Shuffle Dataset")
parser.add_argument("-s", "--seed", type=int, default='0', help="Seed of the shuffle")
parser.add_argument("-st", "--saveto", type=str, default='', help="Save result to a folder")
parser.add_argument("-l", "--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("-f", "--fold", type=str, default='1', help="which fold do you want to test (1, 2, 3, 4 or 5)")

# --- Reading the argument configuration
kwargs = vars(parser.parse_args())
print(kwargs)

### Segmentation configuration
path_image = kwargs['path']
nf = kwargs['nfilters']
batch_size = kwargs['batch_size']
epochs = kwargs['epochs']
gpus = kwargs['gpus']
input_shape = kwargs['input_shape']
shuffle = kwargs['shuffle']
seed = kwargs['seed']
path_save = kwargs['saveto']
lr = kwargs['lr']
fold = kwargs['fold']

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# Create the folder where we are going to save the results
if path_save == '':
    path_save = os.getcwd()  + '/results'

createFolder(path_save)

# Caracteristicas de las imagenes
img_rows = int(input_shape.split('x')[0])
img_cols = int(input_shape.split('x')[1])
RGB = 3
input_shape_i = (img_rows,img_cols, RGB)
input_shape_s = (img_rows,img_cols, 1)

# load model (unet, unet_myocardial, doble_unet, doble_unet_4_connections, CapsNet, unet_and_CapsNet_Ariel, unet_and_CapsNet_Andres)
model = unet_4_niveles(input_shape = input_shape_i, nclass=1, fchannel=-1, nf1 = nf)

# Design model of multigpu or single gpu
# This works for every network architecture except doble_unet and doble_unet_4_connections
if (gpus > 1):
    with tf.device('/cpu:0'):
        model = multi_gpu_model(model, gpus=gpus)
        optimizer = Adam(lr=lr)
        model.compile(loss=jaccard_distance, optimizer=optimizer, metrics=[f1,dice, 'mse','mae', 'accuracy'])

else:
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss=jaccard_distance,metrics=[ 'mse','mae', 'accuracy', f1, dice])

# This works only for doble_unet
"""
mse_capa_4 = Weighted_MSE(1e4)
if (gpus > 1):
    with tf.device('/cpu:0'):
        model = multi_gpu_model(model, gpus=gpus)
        optimizer = Adam(lr=lr)
        model.compile(loss=[jaccard_distance, mse_capa_4], optimizer=optimizer, metrics={'CAPA_5': 'mse', 'UNET': [f1, dice, 'mse','mae', 'acc']})
else:
    optimizer = Adam(lr=lr)
    model.compile(loss=[jaccard_distance, mse_capa_4], optimizer=optimizer, metrics={'CAPA_5': 'mse', 'UNET': [f1, dice, 'mse','mae', 'acc']})
"""

# This works only for doble_unet_4_connections
"""
mse_capa_1 = Weighted_MSE(1e1)
mse_capa_2 = Weighted_MSE(1e2)
mse_capa_3 = Weighted_MSE(1e3)
mse_capa_4 = Weighted_MSE(1e4)

if (gpus > 1):
    model = multi_gpu_model(model, gpus=gpus)
    optimizer = Adam(lr=lr)
    model.compile(loss={'UNET': jaccard_distance, 'Capa1': mse_capa_1, 'Capa2': mse_capa_2, 'Capa3': mse_capa_3, 'Capa4': mse_capa_4}, optimizer=optimizer, metrics={'UNET': [f1, dice, 'mse','mae', 'acc']})

else:
    optimizer = Adam(lr=lr)
    model.compile(loss={'UNET': jaccard_distance, 'Capa1': mse_capa_1, 'Capa2': mse_capa_2, 'Capa3': mse_capa_3, 'Capa4': mse_capa_4}, optimizer=optimizer, metrics={'UNET': [f1, dice, 'mse','mae', 'acc']})
"""

# Let's look how the model is and how much parameters has
model.summary()

from keras.utils import plot_model
plot_model(model, to_file= path_save + '/architecture_graph.png', show_shapes = True)

# This functions load the names of the images files and their segmentation
(x_train, y_train), (x_test, y_test) = load_data_names_nuclei(path = path_image, shuffle=False, seed = None, fold = fold)

# Generators
# Use DataGenerator for every net except doble_unet and doble_unet_4_connections
training_generator = DataGenerator(x_train, dim = (img_rows, img_cols), batch_size = batch_size, n_channels= RGB, shuffle= True)
validation_generator = DataGenerator(x_test,  dim = (img_rows, img_cols), batch_size = batch_size, n_channels= RGB, shuffle= True)

# DataGenerator for doble_unet:
"""
training_generator = DataGenerator_doble_unet(x_train, dim = (img_rows, img_cols), batch_size = batch_size, n_channels= RGB, shuffle= True)
validation_generator = DataGenerator_doble_unet(x_test, dim = (img_rows, img_cols), batch_size = batch_size, n_channels= RGB, shuffle= True)
"""
# DataGenerator for doble_unet_4_connections:
"""
training_generator = DataGenerator_doble_unet_4_connections(x_train, dim = (img_rows, img_cols), batch_size = batch_size, n_channels= RGB, shuffle= True)
validation_generator = DataGenerator_doble_unet_4_connections(x_test, dim = (img_rows, img_cols), batch_size = batch_size, n_channels= RGB, shuffle= True)
"""

# Train model on dataset
history = model.fit_generator(generator = training_generator,
                    validation_data = validation_generator,
                    epochs = epochs,
                    verbose = 1,
                    use_multiprocessing=False,
                    workers=0,
                    shuffle=False)


# This is to not overwrite past files saved
x=0
file_name = ''
while True:
    file_name = 'model_' + str(x)
    exists = os.path.isfile( path_save + '/' + file_name + '.h5')
    if not exists:
        model.save(path_save + '/' + file_name + '.h5')
        np.save(path_save + '/' + file_name + '_history.npy', history.history)
        break
    else:
        x+=1

sys.exit("Finish")
