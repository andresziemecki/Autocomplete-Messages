import numpy as np
import keras
import sys
import os
#from keras import utils.Sequence

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://keras.io/models/sequential/

#function that load the data names
def load_data_names_nuclei(path = '', shuffle=False, seed = None, fold = '1'):

    if path == '':
        sys.exit('Path where image are not given')

    # Read directories of each folder
    DirList = [f for f in os.listdir(path)]

    n = len(DirList) # Esta variable me dice cuantas carpetas va loadiando del total

    # Variables donde vamos a poner todos los datos
    im_train = []
    im_test = []
    ims_train = []
    ims_test = []

    total_de_imagenes = 0

    for it, Dir in enumerate(DirList):

        print("Loading Dataset: " + Dir + " " + str(it) + "/" + str(n), flush=True)

        # Defino los directorios donde estaran las imagenes y su segmentacion
        path_i_np = path + '/' + Dir + '/images_cropped_np'
        path_s_np = path + '/' + Dir + '/segmentation_cropped_np'

        fileList_i = [i for i in os.listdir(path_i_np) if i.endswith('.npy')];
        fileList_s = [s for s in os.listdir(path_s_np) if s.endswith('.npy')];

        if (len(fileList_i) != len(fileList_s)):
            print ("Error, la cantidad de imagenes no es igual a la cantidad segmentada: " + str(len(fileList_i)) + "!=" + str(len(fileList_s)), flush=True)
            break;

        print ("The len of this folder is: {}".format(len(fileList_i)), flush=True)

        total_de_imagenes = total_de_imagenes + len(fileList_i)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(fileList_i)
            seed = seed + 1

        for k,file in enumerate(fileList_i):
            I = path_i_np + '/' + file # Variable auxiliar para las imagenes
            Is = path_s_np + '/' + file # Y para la segmentada
            if (file[0:3] == (fold*3) and Dir == 'nuclei'):
                im_test.append(I)
                ims_test.append(Is)
            else:

                im_train.append(I)
                ims_train.append(Is)

    # Mezclo los nombres para mejorar la robustez de la red
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(im_train)
        seed = seed + 1
        np.random.seed(seed)
        np.random.shuffle(im_test)

    print ("La cantidad total de imagenes son: {}".format(total_de_imagenes), flush = True)
    print ("La cantidad de train son: {}".format(len(im_train)), flush = True)
    print ("La cantidad test son: {}".format(len(im_test)), flush = True)
    # Separo lo que es train de lo que es test en un 70% y 30%
    return (im_train, ims_train), (im_test, ims_test)

# generate images on the fly in the model.fit (in the training)
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=22, dim=(256,256), n_channels=3, shuffle=True, seed=1):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs # Lista de archivos a leer
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.seed = seed
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X1, y1 = self.__data_generation(list_IDs_temp)

        return [X1], [y1]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        """if self.shuffle == True:
            np.random.shuffle(self.indexes)"""
        if self.shuffle == True:
            if self.seed is not None:
                self.seed += 1
                np.random.seed(self.seed)
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # input
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels), dtype = 'float32')
        # output
        y1 = np.empty((self.batch_size, *self.dim, 1), dtype='uint8')
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X1[i,] = np.load(ID)
            # Store class
            y1[i,] = np.load(ID.replace('images_cropped_np', 'segmentation_cropped_np'))
        return X1, y1

class DataGenerator_doble_unet(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=22, dim=(256,256), n_channels=3, shuffle=True, seed=1):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs # Lista de archivos a leer
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.seed = seed
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        [X1, X2], [y1,y2] = self.__data_generation(list_IDs_temp)

        return [X1, X2], [y1,y2]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        """if self.shuffle == True:
            np.random.shuffle(self.indexes)"""
        if self.shuffle == True:
            if self.seed is not None:
                self.seed += 1
                np.random.seed(self.seed)
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # First input
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels), dtype = 'float32')
        # Second input
        X2 = np.empty((self.batch_size, *self.dim, 1), dtype='uint8')
        # First output
        y1 = np.empty((self.batch_size, *self.dim, 1), dtype='uint8')
        # Second output
        y2 = np.zeros((self.batch_size, *(8,8), 512), dtype='uint8')
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X1[i,] = np.load(ID)
            X2[i,] = np.load(ID.replace('images_cropped_np', 'segmentation_cropped_np'))
            # Store class
            y1[i,] = np.load(ID.replace('images_cropped_np', 'segmentation_cropped_np'))

        return [X1, X2], [y1,y2]

class DataGenerator_doble_unet_4_connections(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=22, dim=(256,256), n_channels=3, shuffle=True, seed=1):#, labels
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs # Lista de archivos a leer
        #self.labels = labels # Lista de archivos segmentados
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.seed = seed
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        [X1, X2], [y1,y2,y3,y4,y5] = self.__data_generation(list_IDs_temp)

        return [X1, X2], [y1,y2,y3,y4,y5]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        """if self.shuffle == True:
            np.random.shuffle(self.indexes)"""
        if self.shuffle == True:
            if self.seed is not None:
                self.seed += 1
                np.random.seed(self.seed)
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels), dtype = 'float32')
        X2 = np.empty((self.batch_size, *self.dim, 1), dtype='uint8')
        y1 = np.empty((self.batch_size, *self.dim, 1), dtype='uint8')
        y2 = np.zeros((self.batch_size, *(64,64), 64), dtype='uint8')
        y3 = np.zeros((self.batch_size, *(32,32), 128), dtype='uint8')
        y4 = np.zeros((self.batch_size, *(16,16), 256), dtype='uint8')
        y5 = np.zeros((self.batch_size, *(8,8), 512), dtype='uint8')
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X1[i,] = np.load(ID)
            X2[i,] = np.load(ID.replace('images_cropped_np', 'segmentation_cropped_np'))
            # Store class
            y1[i,] = np.load(ID.replace('images_cropped_np', 'segmentation_cropped_np'))
        return [X1, X2], [y1,y2,y3,y4,y5]
