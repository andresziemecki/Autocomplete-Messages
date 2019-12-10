from __future__ import print_function

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

import lib.io as io
import lib.Architectures as models

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

# Create the folder where we are going to save the results
if path_save == '':
    path_save = os.getcwd()  + '/results'

base_path_save_url = path_save + "_ACNN_"
base_path_encoder = path_save + "_encoder_"
i=0
path_save = base_path_save_url + str(i)
while(os.path.exists(base_path_save_url + str(i))):
    i = i + 1
    path_save = base_path_save_url + str(i)

path_encoder = base_path_encoder + str(i)

io.createFolder(base_path_save_url+ str(i))

f = open(os.path.join(path_save, "command_executed.txt"),"w+")
for s in kwargs.keys():
    f.write(s + ': ' + str(kwargs[s]) + '\t')
f.close()

# Caracteristicas de las imagenes
img_rows = int(input_shape.split('x')[0])
img_cols = int(input_shape.split('x')[1])
RGB = 3
input_shape_i = (img_rows,img_cols, RGB)
input_shape_s = (img_rows,img_cols, 1)

print("\nLoading the encoder model...\n")

encoder_model = load_model(os.path.join(path_encoder, "encoder_model.h5"),
        custom_objects={'jaccard_distance':models.jaccard_distance,
            'dice':models.dice, 'f1':models.f1})

# We have to compile this model loaded before manipulate it because every layer is not trainable. If we don't do this we will get an error warning
if (gpus > 1):
    with tf.device('/cpu:0'):
        encoder_model = multi_gpu_model(encoder_model, gpus=gpus)
optimizer = Adam(lr=lr)
encoder_model.compile(optimizer=optimizer,
        loss=models.jaccard_distance,metrics=[ 'mse','mae', 'accuracy',
            models.f1, models.dice])


"""
Let's build the model for training
"""

print("\nLoading the model for training...\n")
# This model is the U-Net + the encoder part ot the U-Net autoencoder (Encoder_model)
# load model unet for prior training
model = models.ACNN_ariel(input_shape = input_shape_i, encoder_model=encoder_model, nclass=1, fchannel=-1, nf1 = nf)

print("Summary of the model:\n")

model.summary()
# Save and see the model
from tensorflow.keras.utils import plot_model
plot_model(model, to_file= path_save + '/ACNN_architecture_graph.png', show_shapes = True)

# Design model of multigpu or single gpu
# Compile the model with multigpu or single gpu
if (gpus > 1):
    with tf.device('/cpu:0'):
        model = multi_gpu_model(model, gpus=gpus)
optimizer = Adam(lr=lr)
model.compile(loss=[models.jaccard_distance, models.euclidean_distance_loss], 
        optimizer=optimizer, loss_weights=[1., 1.], 
        metrics={'decode_Activation_unet': [models.f1, models.dice, 'mse','mae',
            'acc', models.jaccard_distance], 'salida': ['mse', 'acc', models.euclidean_distance_loss]})

print("\nLoading data names...\n")
# This functions load the names of the images files and their segmentation
(x_train, x_test) = io.load_data_names_nuclei(path = path_image, shuffle=False, seed = seed, fold = fold)

# Generators
training_generator = io.DataGenerator(x_train, dim = (img_rows, img_cols), batch_size = batch_size, n_channels= RGB, shuffle= True)
validation_generator = io.DataGenerator(x_test,  dim = (img_rows, img_cols), batch_size = batch_size, n_channels= RGB, shuffle= True)

print("\nStarting the training of the model...\n")

# Train model on dataset
history = model.fit_generator(generator = training_generator,
                    validation_data = validation_generator,
                    epochs = epochs,
                    verbose = 2,
                    use_multiprocessing=False,
                    workers=0,
                    shuffle=False)


print("\nFinalization of the training model and saving the model trained and the history... \n")

model.save(path_save + '/ACNN.h5')
np.save(path_save + '/ACNN_history.npy', history.history)

print("Trimming the model to get only the unet and save it.\n")

# Trim the model to get the unet trained
layer_names = [layer.name for layer in model.layers]
layer_idx = layer_names.index( 'decode_Activation_unet')
model = Model(model.input[0], model.layers[layer_idx].output)


# We have to compile this model loaded before evaluate it
if (gpus > 1):
    with tf.device('/cpu:0'):
        model = multi_gpu_model(model, gpus=gpus)
optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer,
        loss=models.jaccard_distance,metrics=[ 'mse','mae', 'accuracy',
            models.f1, models.dice])

# Evaluate the unet obtained
print("Evaluation of UNet from ACNN")
evaluate_generator = io.DataUnetGenerator(x_test,  dim = (img_rows, img_cols), batch_size = batch_size, shuffle= True)
print(model.evaluate_generator(generator=evaluate_generator, verbose = 2))

# Plotting the model
plot_model(model, to_file= path_save + '/unet_ACNN_graph.png', show_shapes = True)
model.save(path_save + '/unet_ACNN.h5')
