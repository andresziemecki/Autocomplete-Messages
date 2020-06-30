import imageio
import numpy as np
import tensorflow as tf
from matplotlib.image import imread
import lib.Architectures as models
from tensorflow.keras.models import load_model

# Insert paths
pathToImage = "/home/andres/Documents/TESIS/results/results/imagen HAR.tif"
pathToUnetModel = "/home/andres/Documents/TESIS/results/results/results_unet_1/Unet.h5"
pathToDobleUnetModel = "/home/andres/Documents/TESIS/results/results/results_doble_unet_1/doble_unet.h5"

# Load Image
I = imread(pathToImage).astype('uint8')

# Resize image to a multiple of 64 and save this new image resize that we are going to use to predict 
# Resize I mean to fill with zeros the contours
newImageResized = np.zeros((I.shape[0] + 64-I.shape[0]%64, I.shape[1] + 64-I.shape[1]%64, I.shape[2])).astype('uint8')
newImageResized[:I.shape[0],:I.shape[1],:I.shape[2]] = I
imageio.imwrite('Image_To_Predict.png', newImageResized)

## Now normalize the Image for prediction
I = I.astype('float32')
I -= I.mean(axis=(0,1))[np.newaxis, np.newaxis]
I /= I.std(axis=(0,1))[np.newaxis, np.newaxis] + 1e-7
newImageResized = np.zeros((I.shape[0] + 64-I.shape[0]%64, I.shape[1] + 64-I.shape[1]%64, I.shape[2])).astype('float32')
newImageResized[:I.shape[0],:I.shape[1],:I.shape[2]] = I

# Crop the Image_To_Predict into 64x64 
Lista = list()
for i in range((int(newImageResized.shape[0]/64))):
    for j in range((int(newImageResized.shape[1]/64))):
        Lista.append(newImageResized[j*64:j*64+64,i*64:i*64+64,:])
tmp = np.array(Lista)
images = np.stack(Lista[:])

# Load the models for predictions
unetModel = load_model(pathToUnetModel,
        custom_objects={
            'jaccard_distance':models.jaccard_distance,
            'dice':models.dice,
            'f1':models.f1,
            }
        )

dobleUnetModel = load_model(pathToDobleUnetModel,
        custom_objects={
            'jaccard_distance':models.jaccard_distance,
            'dice':models.dice,
            'f1':models.f1,
            'mean_squared_error_1e4':models.mean_squared_error_1e4,
            }
        )

# Unet Prediction
unetPrediction = unetModel.predict(tmp)

# Doble Unet Prediction
zeros = np.zeros((tmp.shape[0], 64, 64, 1), dtype='float32')
dobleUnetPrediction, _ = dobleUnetModel.predict((tmp, zeros ))

# Reconstruction of the image

#Unet
reconstruccionUnet = np.zeros((newImageResized.shape[0], newImageResized.shape[1], 1) , dtype = 'float32')
step = (int(newImageResized.shape[0]/64))
for i in range((int(newImageResized.shape[0]/64))):
    for j in range((int(newImageResized.shape[1]/64))):
        reconstruccionUnet[j*64:j*64+64,i*64:i*64+64,:]=unetPrediction[j+step*i,...]

imageio.imwrite('Segmentation_Unet_Predicted.png', reconstruccionUnet)

# DobleUnet
reconstruccionDobleUnet = np.zeros((newImageResized.shape[0], newImageResized.shape[1], 1) , dtype = 'float32')
step = (int(newImageResized.shape[0]/64))
for i in range((int(newImageResized.shape[0]/64))):
    for j in range((int(newImageResized.shape[1]/64))):
        reconstruccionDobleUnet[j*64:j*64+64,i*64:i*64+64,:]=dobleUnetPrediction[j+step*i,...]

imageio.imwrite('Segmentation_DobleUnet_Predicted.png', reconstruccionDobleUnet)