import numpy as np
import matplotlib.pyplot as plt
import PIL
import skimage
import sys
import os
import argparse

from PIL import Image
from skimage import measure
from skimage import exposure
from scipy import ndimage

#---- Creamos los argumentos que va a recibir desde afuera
parser = argparse.ArgumentParser()

parser.add_argument('-p', '--path', type=str, default= '', help='Path where image and segmentations are. Images should be in a folder called "images" and segmentation in a folder called "segmentation". Files in each folder (images and their segmnetation) should have the same name, but not necessary the same extension')
parser.add_argument("-st", "--saveto", type=str, default='', help="Path where to save cropped images. It will create two new folders called images_cropped and segmentation_cropped. Default: the same folder where images and segmentation are.")
parser.add_argument("-cr", "--crop_shape", type=str, default='64x64', help="Input shape from de Dataset")

# --- Reading the argument configuration
kwargs = vars(parser.parse_args())
print(kwargs)

path = kwargs['path']
if path == '':
    sys.exit('No path given')
if path[-1] == '/':
    path = path[:-1]
folder_save = kwargs['saveto']
if folder_save == '':
    folder_save = path
# Shape of images and segmentation that it will be cropped
crop_shape = kwargs['crop_shape']
img_shape_i, img_shape_j = [int(f) for f in crop_shape.split('x')]


# Folders where I'll took the images and segmentations
path_i = path +'/images'
path_s = path + '/segmentation'

# Folders where I'll save images and segmentations
path_save_i = folder_save + '/images_cropped_np';
path_save_s = folder_save + '/segmentation_cropped_np';

# Create folder function, if exits, don't create anything
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# Function that normalice an image and his segmentation
def normalice(i,s):
    if (len(i.shape) == 2):
        i = i[..., np.newaxis]
        i = np.repeat(i, 3, axis=2)
    i -= i.mean(axis=(0,1))[np.newaxis, np.newaxis]
    i /= i.std(axis=(0,1))[np.newaxis, np.newaxis] + 1e-7
    s = np.uint8(s > 0)[..., np.newaxis]
    return i,s

# Create both folders to save images and segmentation
createFolder(path_save_i);
createFolder(path_save_s);

# Get the file List images
fileList_i = [f for f in os.listdir(path_i) if (f.lower().endswith('jpg') or f.lower().endswith('png') or f.lower().endswith('bmp') or f.lower().endswith('tif'))];
print("Number of images: " + str(len(fileList_i)) )

# Get the file List segmentation
fileList_s = [s for s in os.listdir(path_s) if (s.lower().endswith('jpg') or s.lower().endswith('png') or f.lower().endswith('bmp') or f.lower().endswith('tif'))];
print("Number of segmentation images: " + str(len(fileList_s)) )

if (len(fileList_i) != len(fileList_s)):
    sys.exit("Number of images and segmentation are not equal")

# Sort both list to get images and his segmentation correctly
fileList_i.sort()
fileList_s.sort()

# Check if the both list have the same name
for i in range(len(fileList_i)):
    if fileList_i[i].split('.')[0] != fileList_s[i].split('.')[0]:
        sys.exit(fileList_i[i].split('.')[0] + " and " + fileList_s[i] + " Not the same name")

step_to_print = int(len(fileList_i)/10)

# In each file of image do the next
for it,file in enumerate(fileList_i):
    if (it%step_to_print == 0):
        print('#', end = '', flush = True)

    # Get the segmentation image according to the image that we just have read
    file_s = fileList_s[fileList_i.index(file)]

    # Open both, image and his segmentation
    I = Image.open(path_i + '/' + file);
    Is = Image.open(path_s + '/' + file_s);

    # transform it to numpy array
    im = np.array(I).astype('float32')
    ims = np.array(Is).astype('float32')

    # normalice those images and segmentation
    im, ims = normalice (im, ims)

    # To each segmented nuclei, asign a number
    ims_labeled = measure.label(ims[..., 0])

    # We will have as a output the true coordenates of the segmentation
    r,c = np.where(ims[..., 0])

    # We'll take the mean of this true pixels of the images pixel and do the mean
    mean = im[r,c,0].mean()

    # Now we are generating the threshold image
    im_th = im[...,0] > mean #image with threshold

    # Because our threshold image is white where is true, we need to invert to fill binary holes in the next line
    im_th_invert = np.invert(im_th)

    #  Filling binary holes
    im_th_invert_fill = ndimage.morphology.binary_fill_holes(im_th_invert)

    # number of images that we are going to do for each image, same number of segmentation nuclei
    cantidad_etiquetas = ims_labeled.max()

    # We need to define the size of the original image because not all image have the same size
    size_i, size_j = im.shape[0], im.shape[1]

    # We don't want to crop outside of the image. So we have to define the max number of the position that we can crop
    i_max = size_i - img_shape_i
    j_max = size_j - img_shape_j

    # Algorithm that it will crop the images
    for k in range(cantidad_etiquetas):
        tmp_i, tmp_j = np.where(ims_labeled == k)
        i = np.amin(tmp_i)
        j = np.amin(tmp_j)
        # We have to check the condition if it is near to the end of the image
        #im[fila,columna,color]
        if (i <= i_max and j <= j_max ):
            im_crop = im[i:(i+img_shape_i),j:(j+img_shape_j), ...];
            ims_crop = ims[i:(i+img_shape_i),j:(j+img_shape_j), ...];
        elif ((i > i_max) and (j > j_max)):
            im_crop = im[i_max:size_i,j_max:size_j,  ...];
            ims_crop = ims[i_max:size_i,j_max:size_j,  ...];
        elif (i > i_max):
            im_crop = im[i_max:size_i, j:(j+img_shape_j), ...];
            ims_crop = ims[i_max:size_i, j:(j+img_shape_j), ...];
        else: #(j > j_max)
            im_crop = im[i:(i+img_shape_i), j_max:size_j, ... ];
            ims_crop = ims[i:(i+img_shape_i), j_max:size_j, ... ];

        # Save those Images cropped
        if (k%5 == 0):
            np.save(path_save_i + '/' +'111_'+ file.split('.')[0] + '_' + str(i) + '_' + str(j), im_crop)
            np.save(path_save_s + '/' +'111_'+ file.split('.')[0] + '_' + str(i) + '_' + str(j), ims_crop)
        elif (k%5 == 1):
            np.save(path_save_i + '/' +'222_'+ file.split('.')[0] + '_' + str(i) + '_' + str(j), im_crop)
            np.save(path_save_s + '/' +'222_'+ file.split('.')[0] + '_' + str(i) + '_' + str(j), ims_crop)
        elif (k%5 == 2):
            np.save(path_save_i + '/' +'333_'+ file.split('.')[0] + '_' + str(i) + '_' + str(j), im_crop)
            np.save(path_save_s + '/' +'333_'+ file.split('.')[0] + '_' + str(i) + '_' + str(j), ims_crop)
        elif (k%5 == 3):
            np.save(path_save_i + '/' +'444_'+ file.split('.')[0] + '_' + str(i) + '_' + str(j), im_crop)
            np.save(path_save_s + '/' +'444_'+ file.split('.')[0] + '_' + str(i) + '_' + str(j), ims_crop)
        else:
            np.save(path_save_i + '/' +'555_'+ file.split('.')[0] + '_' + str(i) + '_' + str(j), im_crop)
            np.save(path_save_s + '/' +'555_'+ file.split('.')[0] + '_' + str(i) + '_' + str(j), ims_crop)

    # Now we want to geenrate images without any nuclei inside
    im_th_invert_fill = np.uint8(im_th_invert_fill)
    # This is the algorithm to do that
    x,y=(0,0)
    im_th_crop = np.array([])
    imagen = np.array([])
    iteracion = 0
    while (x<= i_max):
        while (y<=j_max):
            im_th_crop = im_th_invert_fill[x:(x+img_shape_i),y:(y+img_shape_j)]
            if (im_th_crop.sum() == 0):
                imagen = im[x:(x+img_shape_i),y:(y+img_shape_j),...]
                imagen_s = ims[x:(x+img_shape_i),y:(y+img_shape_j), ...]
                if (iteracion%5 == 0):
                    np.save(path_save_i + '/'+'111_' + file.split('.')[0] + '_nothing_x_' + str(x) + '_y_' + str(y), imagen)
                    np.save(path_save_s + '/'+'111_' + file.split('.')[0] + '_nothing_x_' + str(x) + '_y_' + str(y), imagen_s)
                elif (iteracion%5 == 1):
                    np.save(path_save_i + '/'+'222_' + file.split('.')[0] + '_nothing_x_' + str(x) + '_y_' + str(y), imagen)
                    np.save(path_save_s + '/'+'222_' + file.split('.')[0] + '_nothing_x_' + str(x) + '_y_' + str(y), imagen_s)
                elif (iteracion%5 == 2):
                    np.save(path_save_i + '/'+'333_' + file.split('.')[0] + '_nothing_x_' + str(x) + '_y_' + str(y), imagen)
                    np.save(path_save_s + '/'+'333_' + file.split('.')[0] + '_nothing_x_' + str(x) + '_y_' + str(y), imagen_s)
                elif (iteracion%5 == 3):
                    np.save(path_save_i + '/'+'444_' + file.split('.')[0] + '_nothing_x_' + str(x) + '_y_' + str(y), imagen)
                    np.save(path_save_s + '/'+'444_' + file.split('.')[0] + '_nothing_x_' + str(x) + '_y_' + str(y), imagen_s)
                else:
                    np.save(path_save_i + '/'+'555_' + file.split('.')[0] + '_nothing_x_' + str(x) + '_y_' + str(y), imagen)
                    np.save(path_save_s + '/'+'555_' + file.split('.')[0] + '_nothing_x_' + str(x) + '_y_' + str(y), imagen_s)
                y+= img_shape_j
                iteracion = iteracion + 1
            else:
                y+=1
        x+=img_shape_i
print('\nfinish')
