import os
import argparse
import numpy as np
import tensorflow as tf


from tensorflow.keras.optimizers import Adam
import lib.io as io
import lib.Architectures as models
from tensorflow.keras.utils import multi_gpu_model

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

# Create the folder where we are going to save the results
if path_save == '':
    path_save = os.getcwd()  + '/results'

base_path_save_url = path_save + "_doble_unet_"

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
input_shape_s = (img_rows,img_cols, 1)

# load model 
model = models.doble_unet(input_shape = input_shape_i, nclass=1, fchannel=-1, nf1 = nf)

# Design model of multigpu or single gpu

if (gpus > 1):
    with tf.device('/cpu:0'):
        model = multi_gpu_model(model, gpus=gpus)
optimizer = Adam(lr=lr)
model.compile(loss=[models.jaccard_distance, models.mean_squared_error_1e4], optimizer=optimizer, metrics={'Capa5': 'mse', 'UNET': [models.f1, models.dice, 'mse','mae', 'acc']})

# Let's look how the model is and how much parameters has
model.summary()

# This functions load the names of the images files and their segmentation
(x_train, x_test) = io.load_data_names_nuclei(path = path_image, shuffle=False, seed = None, fold = fold)


# Generators

training_generator = io.DataGenerator_doble_unet(x_train, dim = (img_rows, img_cols), batch_size = batch_size, n_channels= RGB, shuffle= True)
validation_generator = io.DataGenerator_doble_unet(x_test, dim = (img_rows, img_cols), batch_size = batch_size, n_channels= RGB, shuffle= True)

# Train model on dataset
history = model.fit_generator(generator = training_generator,
                    validation_data = validation_generator,
                    epochs = epochs,
                    verbose = 2,
                    use_multiprocessing=False,
                    workers=0,
                    shuffle=False)


model.save(path_save + '/doble_unet.h5')
np.save(path_save + '/doble_unet_history.npy', history.history)

