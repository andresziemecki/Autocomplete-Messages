import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import lib.Architectures as models
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam
import lib.io as io

parser = argparse.ArgumentParser()

parser.add_argument("-st", "--saveto", type=str, default='', help="Save result to a folder, default = ./results")
parser.add_argument("-g", "--gpus", type=int, default=1, help="# of GPUs to use for training (default = 1)")
parser.add_argument('-p', '--path', type=str, default='', help='Path where the dataset is')
parser.add_argument("-is", "--input_shape", type=str, default='64x64', help="Input shape from de Dataset")
parser.add_argument('-bs', '--batch_size', type=int, default=72, help='Batch size (default: 32)')
parser.add_argument("-f", "--fold", type=str, default='1', help="which fold do you want to test (1, 2, 3, 4 or 5)")

kwargs = vars(parser.parse_args())

batch_size = kwargs['batch_size']
input_shape = kwargs['input_shape']
path_image = kwargs['path']
path_save = kwargs['saveto']
gpus = kwargs['gpus']
fold = kwargs['fold']

# Create the folder where we are going to save the results
if path_save == '':
    path_save = os.getcwd()  + '/results'

base_path_save_url = path_save + "_ACNN_"
i=0
path_save = base_path_save_url + str(i)
while(os.path.exists(base_path_save_url + str(i))):
    i = i + 1

path_model = base_path_save_url + str(i-1)

path_model = path_model + '/unet_ACNN.h5'
model = load_model(path_model,
        custom_objects={'jaccard_distance':models.jaccard_distance,
            'dice':models.dice, 'f1':models.f1})


# We have to compile this model loaded before evaluate it
if (gpus > 1):
    with tf.device('/cpu:0'):
        model = multi_gpu_model(model, gpus=gpus)
optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer,
        loss=models.jaccard_distance,metrics=[ 'mse','mae', 'accuracy',
            models.f1, models.dice])

# Evaluate the unet obtained
img_rows = int(input_shape.split('x')[0])
img_cols = int(input_shape.split('x')[1])
(x_train, x_test) = io.load_data_names_nuclei(path = path_image, shuffle=False, seed = None, fold = fold)
evaluate_generator = io.DataUnetGenerator(x_test,  dim = (img_rows, img_cols), batch_size = batch_size, shuffle= True)
print(model.evaluate_generator(generator=evaluate_generator, verbose = 1))