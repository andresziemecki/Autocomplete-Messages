tesis-andres-cell

*****************************************
crop_images.py

has 3 arguments:

--path=

Path where images and their segmentation are.
Images should be in a folder called "images" and segmentation in a folder called "segmentation" (inside the path).
Files in each folder (images and their segmentation) should have the same name, but not necessary the same extension.

--saveto=

Path to save cropped images.
It will create two new folders called "images_cropped" and "segmentation_cropped".
Default: the same folder where images and their segmentation are.

--crop_shape=

Size of crop shape. It should be passed like this: 64x64
Default: 64x64

******************************************
lib/Arquitectures.py

File with every Architecture that has been used for nuclei segmentation
Also there are some losses functions as: jaccard_distance f1 dice and Weighted_MSE

unet_myocardial
presented in the paper: Automatic myocardial segmentation by using a deep learning network in cardiac MRI

unet
presented in the paper: U-Net: Convolutional Networks for Biomedical Image Segmentation

doble_unet
net selected for the solution of nuclei segmentation in the work of Ariel Curiale and Andres Ziemecki

doble_unet_4_connections
net similar to doble_unet but instead of the connection in the last layer, it has 4 connections, one in each level.

CapsNet
presented in the paper: Dynamic Routing between Capsules - by Hinton. It was modified to adapt it to the problem of nuclei segmentation

unet_and_CapsNet_Ariel
parallel combination of unet and caps net (look where it's the concatenation using plot_model from keras.utils)

unet_and_CapsNet_Andres
parallel combination of unet and caps net (look where it's the concatenation using plot_model from keras.utils)

****************************************************
lib/io.py

File where it produce the streaming while the training is happening.
It has 1 function and 3 classes.

function: "load_data_names_nuclei" which load the names of each file for training set and test set
clases:
"DataGenerator" Generates the images on the fly while training for every architecture except for doble_unet and doble_unet_4_connections
"DataGenerator_doble_unet" Generates the images on the fly while training for doble_unet
"DataGenerator_doble_unet_4_connections" Generates the images on the fly while training for doble_unet_4_connections


****************************************************
train.py

File that you can modify by your own.
Be careful! See that different architectures have different compile functions and DataGenerator
The file receives a lot of parameters, see parser function
