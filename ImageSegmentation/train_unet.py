from __future__ import print_function

import os
import argparse
import numpy as np
import tensorflow as tf
# Set Seeds for repetibility
# Importante que vaya antes del import de la arquitectura que carga keras
np.random.seed(1)
tf.random.set_seed(2)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.utils import plot_model

import lib.io as io
import lib.Architectures as models


#----- Parche de Tensorflow 2.0 para mi gPU
#  from tensorflow.compat.v1 import ConfigProto
#  from tensorflow.compat.v1 import InteractiveSession
#  config = ConfigProto()
#  config.gpu_options.per_process_gpu_memory_fraction = 0.9
#  config.gpu_options.allow_growth = True
#  session = InteractiveSession(config=config)
#----- Parche




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
parser.add_argument("-st", "--saveto", type=str, default='', help="Save result to a folder, default = ./results")
parser.add_argument("-l", "--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("-f", "--fold", type=str, default='1', help="which fold do you want to test (1, 2, 3, 4 or 5)")
parser.add_argument("-dtr", "--dataset_train", type=str, default='all', help="which dataset train (2009_ISBI_2DNuclei, BBBC018, nuclei, nucleisegmentationbenchmark)")
parser.add_argument("-dte", "--dataset_test", type=str, default='nuclei', help="which dataset test")

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
dataset_train = kwargs['dataset_train']
dataset_test = kwargs['dataset_test']

# Create the folder where we are going to save the results
if path_save == '':
    path_save = os.getcwd()  + '/results'

base_path_save_url = path_save + "_unet_"

path_save = base_path_save_url + str(fold)

io.createFolder(base_path_save_url+ str(fold))

f = open(os.path.join(path_save, "command_executed.txt"),"w+")
for s in kwargs.keys():
    f.write(s + ': ' + str(kwargs[s]) + '\t')
f.close()

# Caracteristicas de las imagenes
img_rows = int(input_shape.split('x')[0])
img_cols = int(input_shape.split('x')[1])
RGB = 3
input_shape_i = (img_rows,img_cols, RGB)


"""
Let's build the model for training
"""

print("\nLoading the model for training...\n")
# This model is the U-Net + the encoder part ot the U-Net autoencoder (Encoder_model)
# load model unet for prior training
model = models.unet(input_shape=input_shape_i, nclass=1, fchannel=-1, nf1=nf, 
        l2reg=0)

print("Summary of the model:\n")

model.summary()
# Save and see the model
plot_model(model, to_file=path_save+'/Unet_architecture_graph.png', 
        show_shapes=True)

# Design model of multigpu or single gpu
# Compile the model with multigpu or single gpu
if (gpus > 1):
    with tf.device('/cpu:0'):
        model = multi_gpu_model(model, gpus=gpus)
optimizer = Adam(lr=lr)
model.compile(
        optimizer=optimizer, 
        loss=models.jaccard_distance, 
        metrics=[models.dice, 'mse']
        )

print("\nLoading data names...\n")
# This functions load the names of the images files and their segmentation
(x_train, x_test) = io.load_data_names_nuclei(path=path_image, shuffle=False, 
        seed=None, fold=fold, dataset_train=dataset_train, dataset_test=dataset_test)

# Generators
training_generator = io.DataGenerator_unet(x_train, dim=(img_rows, img_cols),
        batch_size=batch_size, n_channels=RGB, shuffle=True)
validation_generator = io.DataGenerator_unet(x_test,  dim=(img_rows, img_cols), 
        batch_size=batch_size, n_channels=RGB, shuffle=True)
        


print("\nStarting the training of the model...\n")

# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    verbose=2,
                    use_multiprocessing=False,
                    workers=0,
                    shuffle=False)


print("\nFinalization of the training model and saving the model trained and the history... \n")

model.save(path_save + '/Unet.h5')
np.save(path_save + '/Unet_history.npy', history.history)
