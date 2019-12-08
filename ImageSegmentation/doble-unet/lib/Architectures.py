import os
import numpy as np
import tensorflow as tf

# Set Seeds for repetibility
np.random.seed(1)
tf.set_random_seed(2)

import keras
import keras.backend as k
from keras import backend as K
from keras.models import Model
from keras.layers import UpSampling2D, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Add, Input, Subtract
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import models, optimizers, initializers, layers

from tensorflow.python.ops.parallel_for import batch_jacobian as tf_batch_jacobian

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def dice(y_true, y_pred):
    '''Calculates the dices's coefficient rate
    between predicted and target values.
    '''
    smooth = 1.
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    intersection = k.sum(y_true_f * y_pred_f)
    d =  (2. * intersection + smooth) / (k.sum(y_true_f) + k.sum(y_pred_f) + smooth)
    return d


def jaccard_distance(y_true, y_pred):
    '''Calculates the Jaccard index
    between predicted and target values.
    '''
    smooth = 1.
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    intersection = k.sum(y_true_f * y_pred_f)
    denom = k.sum(y_true_f) + k.sum(y_pred_f) - intersection
    j =  (intersection + smooth) / (denom + smooth)

    return 1 - j


class Weighted_MSE(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        return self.mean_squared_error_(y_true, y_pred)

    def mean_squared_error_(self, y_true, y_pred):
        return self.alpha*K.mean(K.square(y_pred), axis=-1)

def unet_myocardial(input_shape, nclass=1, fchannel=-1, nf1 = 64):

    """Paper: Automatic myocardial segmentation by using a deep learning network in cardiac MRI"""

    input_layer = Input(input_shape)

    # Layer 1
    rl1 = Conv2D(nf1, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(input_layer)
    rl1 = BatchNormalization(axis=fchannel)(rl1)

    l1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(input_layer)
    l1 = BatchNormalization(axis=fchannel)(l1)
    l1 = Activation('relu')(l1)
    l1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l1)
    l1 = Add()([rl1,l1])
    l1 = Activation('relu')(l1)

    # Layer 2
    l2 = MaxPooling2D(pool_size=(2, 2))(l1)

    nf2 = nf1*2
    rl2 = Conv2D(nf2, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    rl2 = BatchNormalization(axis=fchannel)(rl2)

    l2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    l2 = BatchNormalization(axis=fchannel)(l2)
    l2 = Activation('relu')(l2)
    l2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    l2 = Add()([rl2,l2])
    l2 = Activation('relu')(l2)

    # Layer 3
    l3 = MaxPooling2D(pool_size=(2, 2))(l2)

    nf3 = nf2*2
    rl3 = Conv2D(nf3, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    rl3 = BatchNormalization(axis=fchannel)(rl3)

    l3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    l3 = BatchNormalization(axis=fchannel)(l3)
    l3 = Activation('relu')(l3)
    l3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    l3 = Add()([rl3,l3])
    l3 = Activation('relu')(l3)

    # Layer 4
    l4 = MaxPooling2D(pool_size=(2, 2))(l3)

    nf4 = nf3*2
    rl4 = Conv2D(nf4, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    rl4 = BatchNormalization(axis=fchannel)(rl4)

    l4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    l4 = BatchNormalization(axis=fchannel)(l4)
    l4 = Activation('relu')(l4)
    l4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    l4 = Add()([rl4,l4])
    l4 = Activation('relu')(l4)

    # Layer 5
    l5 = MaxPooling2D(pool_size=(2, 2))(l4)

    nf5 = nf4*2
    rl5 = Conv2D(nf5, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l5)
    rl5 = BatchNormalization(axis=fchannel)(rl5)

    l5 = Conv2D(nf5, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l5)
    l5 = BatchNormalization(axis=fchannel)(l5)
    l5 = Activation('relu')(l5)
    l5 = Conv2D(nf5, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l5)
    l5 = Add()([rl5,l5])
    l5 = Activation('relu')(l5)

    # Layer 4
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up4 = UpSampling2D(size=(2,2))(l5)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv4 = Conv2D(nf4, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up4)
    upconv4 = BatchNormalization(axis=fchannel)(upconv4)
    # End Up-conv

    # ---------- Level 4
    m_concat4 = Concatenate(axis=fchannel)([l4, upconv4])

    drl4 = Conv2D(nf4, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat4)
    drl4 = BatchNormalization(axis=fchannel)(drl4)

    dl4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat4)
    dl4 = BatchNormalization(axis=fchannel)(dl4)
    dl4 = Activation('relu')(dl4)
    dl4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl4)
    dl4 = Add()([drl4,dl4])
    dl4 = Activation('relu')(dl4)

    # Layer 3
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up3 = UpSampling2D(size=(2,2))(dl4)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv3 = Conv2D(nf3, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up3)
    upconv3 = BatchNormalization(axis=fchannel)(upconv3)
    # End Up-conv

    # ---------- Level 3
    m_concat3 = Concatenate(axis=fchannel)([l3, upconv3])

    drl3 = Conv2D(nf3, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat3)
    drl3 = BatchNormalization(axis=fchannel)(drl3)

    dl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat3)
    dl3 = BatchNormalization(axis=fchannel)(dl3)
    dl3 = Activation('relu')(dl3)
    dl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl3)
    dl3 = Add()([drl3,dl3])
    dl3 = Activation('relu')(dl3)

    # Layer 2
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up2 = UpSampling2D(size=(2,2))(dl3)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv2 = Conv2D(nf2, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up2)
    upconv2 = BatchNormalization(axis=fchannel)(upconv2)
    # End Up-conv

    # ---------- Level 2
    m_concat2 = Concatenate(axis=fchannel)([l2, upconv2])

    drl2 = Conv2D(nf2, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat2)
    drl2 = BatchNormalization(axis=fchannel)(drl2)

    dl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat2)
    dl2 = BatchNormalization(axis=fchannel)(dl2)
    dl2 = Activation('relu')(dl2)
    dl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl2)
    dl2 = Add()([drl2,dl2])
    dl2 = Activation('relu')(dl2)

    # Layer 1
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up1 = UpSampling2D(size=(2,2))(dl2)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv1 = Conv2D(nf1, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up1)
    upconv1 = BatchNormalization(axis=fchannel)(upconv1)
    # End Up-conv

    # ---------- Level 1
    m_concat1 = Concatenate(axis=fchannel)([l1, upconv1])

    drl1 = Conv2D(nf1, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat1)
    drl1 = BatchNormalization(axis=fchannel)(drl1)

    dl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat1)
    dl1 = BatchNormalization(axis=fchannel)(dl1)
    dl1 = Activation('relu')(dl1)
    dl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl1)
    dl1 = Add()([drl1,dl1])
    dl1 = Activation('relu')(dl1)

    # Final layer
    decode = Conv2D(nclass, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl1)
    decode = Activation('sigmoid')(decode)

    model = Model(inputs=input_layer, outputs=decode)

    return model


def unet(input_shape, nclass=1, fchannel=-1, nf1 = 64):

    """ Paper: U-Net: Convolutional Networks for Biomedical Image Segmentation"""

    input_layer = Input(input_shape)
    keras.initializers.glorot_uniform(seed=0)

    # Layer 1
    l1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(input_layer)
    l1 = Activation('relu')(l1)
    l1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l1)
    l1 = Activation('relu')(l1)

    # Layer 2
    l2 = MaxPooling2D(pool_size=(2, 2))(l1)

    nf2 = 2*nf1
    l2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    l2 = Activation('relu')(l2)
    l2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    l2 = Activation('relu')(l2)

    # Layer 3
    l3 = MaxPooling2D(pool_size=(2, 2))(l2)

    nf3 = 2*nf2
    l3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    l3 = Activation('relu')(l3)
    l3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    l3 = Activation('relu')(l3)

    # Layer 4

    l4 = MaxPooling2D(pool_size=(2, 2))(l3)

    nf4 = 2*nf3
    l4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    l4 = Activation('relu')(l4)
    l4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    l4 = Activation('relu')(l4)

    # Layer 5

    l5 = MaxPooling2D(pool_size=(2, 2))(l4)

    nf5 = 2*nf4
    l5 = Conv2D(nf5, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l5)
    l5 = Activation('relu')(l5)
    l5 = Conv2D(nf5, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l5)
    l5 = Activation('relu')(l5)

    # Layer 4
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up4 = UpSampling2D(size=(2,2))(l5)
    # Convolution without activation, so, a(x) = x
    upconv4 = Conv2D(nf4, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up4)
    # End Up-conv

    # ---------- Level 4
    m_concat4 = Concatenate(axis=fchannel)([l4, upconv4])

    dl4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat4)
    dl4 = Activation('relu')(dl4)
    dl4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl4)
    dl4 = Activation('relu')(dl4)

    # Layer 3
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up3 = UpSampling2D(size=(2,2))(dl4)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv3 = Conv2D(nf3, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up3)
    # End Up-conv

    # ---------- Level 3
    m_concat3 = Concatenate(axis=fchannel)([l3, upconv3])

    dl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat3)
    dl3 = Activation('relu')(dl3)
    dl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl3)
    dl3 = Activation('relu')(dl3)

    # Layer 2
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up2 = UpSampling2D(size=(2,2))(dl3)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv2 = Conv2D(nf2, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up2)
    #upconv2 = BatchNormalization(axis=fchannel)(upconv2)
    # End Up-conv

    # ---------- Level 2
    m_concat2 = Concatenate(axis=fchannel)([l2, upconv2])

    dl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat2)
    dl2 = Activation('relu')(dl2)
    dl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl2)
    dl2 = Activation('relu')(dl2)

    # Layer 1
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up1 = UpSampling2D(size=(2,2))(dl2)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv1 = Conv2D(nf1, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up1)
    #upconv1 = BatchNormalization(axis=fchannel)(upconv1)
    # End Up-conv

    # ---------- Level 1
    m_concat1 = Concatenate(axis=fchannel)([l1, upconv1])

    dl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat1)
    dl1 = Activation('relu')(dl1)
    dl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl1)
    dl1 = Activation('relu')(dl1)

    # Final layer
    decode = Conv2D(nclass, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl1)
    decode = Activation('sigmoid')(decode)

    model = Model(inputs=input_layer, outputs=decode)

    return model


def unet_4_niveles(input_shape, nclass=1, fchannel=-1, nf1 = 64):


    input_layer = Input(input_shape)
    keras.initializers.glorot_uniform(seed=0)

    # Layer 1
    l1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(input_layer)
    l1 = Activation('relu')(l1)
    l1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l1)
    l1 = Activation('relu')(l1)

    # Layer 2
    l2 = MaxPooling2D(pool_size=(2, 2))(l1)

    nf2 = 2*nf1
    l2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    l2 = Activation('relu')(l2)
    l2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    l2 = Activation('relu')(l2)

    # Layer 3
    l3 = MaxPooling2D(pool_size=(2, 2))(l2)

    nf3 = 2*nf2
    l3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    l3 = Activation('relu')(l3)
    l3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    l3 = Activation('relu')(l3)

    # Layer 4

    l4 = MaxPooling2D(pool_size=(2, 2))(l3)

    nf4 = 2*nf3
    l4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    l4 = Activation('relu')(l4)
    l4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    l4 = Activation('relu')(l4)

    # Layer 3
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up3 = UpSampling2D(size=(2,2))(l4)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv3 = Conv2D(nf3, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up3)
    # End Up-conv

    # ---------- Level 3
    m_concat3 = Concatenate(axis=fchannel)([l3, upconv3])

    dl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat3)
    dl3 = Activation('relu')(dl3)
    dl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl3)
    dl3 = Activation('relu')(dl3)

    # Layer 2
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up2 = UpSampling2D(size=(2,2))(dl3)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv2 = Conv2D(nf2, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up2)
    #upconv2 = BatchNormalization(axis=fchannel)(upconv2)
    # End Up-conv

    # ---------- Level 2
    m_concat2 = Concatenate(axis=fchannel)([l2, upconv2])

    dl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat2)
    dl2 = Activation('relu')(dl2)
    dl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl2)
    dl2 = Activation('relu')(dl2)

    # Layer 1
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up1 = UpSampling2D(size=(2,2))(dl2)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv1 = Conv2D(nf1, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up1)
    #upconv1 = BatchNormalization(axis=fchannel)(upconv1)
    # End Up-conv

    # ---------- Level 1
    m_concat1 = Concatenate(axis=fchannel)([l1, upconv1])

    dl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat1)
    dl1 = Activation('relu')(dl1)
    dl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl1)
    dl1 = Activation('relu')(dl1)

    # Final layer
    decode = Conv2D(nclass, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl1)
    decode = Activation('sigmoid')(decode)

    model = Model(inputs=input_layer, outputs=decode)

    return model

def doble_unet(input_shape, nclass=1, fchannel=-1, nf1 = 64):

    input_layer = Input(input_shape)
    input_segmentation = Input([input_shape[0], input_shape[1], 1])

    keras.initializers.glorot_uniform(seed=0)

    """ Imagen codification """

    # Layer 1
    l1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(input_layer)
    l1 = Activation('relu')(l1)
    l1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l1)
    l1 = Activation('relu')(l1)

    # Layer 2
    l2 = MaxPooling2D(pool_size=(2, 2))(l1)

    nf2 = 2*nf1

    l2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    l2 = Activation('relu')(l2)
    l2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    l2 = Activation('relu')(l2)

    # Layer 3
    l3 = MaxPooling2D(pool_size=(2, 2))(l2)

    nf3 = 2*nf2

    l3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    l3 = Activation('relu')(l3)
    l3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    l3 = Activation('relu')(l3)

    # Layer 4
    l4 = MaxPooling2D(pool_size=(2, 2))(l3)

    nf4 = 2*nf3

    l4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    l4 = Activation('relu')(l4)
    l4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    l4 = Activation('relu')(l4)

    """ Segmentation codification """

    sl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(input_segmentation)
    sl1 = Activation('relu')(sl1)
    sl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(sl1)
    sl1 = Activation('relu')(sl1)

    # Layer 2
    sl2 = MaxPooling2D(pool_size=(2, 2))(sl1)

    nf2 = 2*nf1

    sl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(sl2)
    sl2 = Activation('relu')(sl2)
    sl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(sl2)
    sl2 = Activation('relu')(sl2)

    # Layer 3
    sl3 = MaxPooling2D(pool_size=(2, 2))(sl2)

    nf3 = 2*nf2

    sl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(sl3)
    sl3 = Activation('relu')(sl3)
    sl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(sl3)
    sl3 = Activation('relu')(sl3)

    # Layer 4
    sl4 = MaxPooling2D(pool_size=(2, 2))(sl3)

    nf4 = 2*nf3

    sl4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(sl4)
    sl4 = Activation('relu')(sl4)
    sl4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(sl4)
    sl4 = Activation('relu')(sl4)

    resta = Subtract(name='Capa5')([sl4,l4])

    """ Imagen decodification """
    # Layer 3
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up3 = UpSampling2D(size=(2,2))(l4)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv3 = Conv2D(nf3, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up3)
    # End Up-conv

    # ---------- Level 3
    m_concat3 = Concatenate(axis=fchannel)([l3, upconv3])

    dl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat3)
    dl3 = Activation('relu')(dl3)
    dl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl3)
    dl3 = Activation('relu')(dl3)

    # Layer 2
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up2 = UpSampling2D(size=(2,2))(dl3)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv2 = Conv2D(nf2, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up2)
    # End Up-conv

    # ---------- Level 2
    m_concat2 = Concatenate(axis=fchannel)([l2, upconv2])

    dl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat2)
    dl2 = Activation('relu')(dl2)
    dl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl2)
    dl2 = Activation('relu')(dl2)

    # Layer 1
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up1 = UpSampling2D(size=(2,2))(dl2)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv1 = Conv2D(nf1, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up1)
    # End Up-conv

    # ---------- Level 1
    m_concat1 = Concatenate(axis=fchannel)([l1, upconv1])

    dl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat1)
    dl1 = Activation('relu')(dl1)
    dl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl1)
    dl1 = Activation('relu')(dl1)

    # Final layer
    decode = Conv2D(nclass, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl1)
    decode = Activation('sigmoid', name='UNET')(decode)

    model = Model(inputs=[input_layer,input_segmentation], outputs=[decode,resta])

    return model

def doble_unet_4_connections(input_shape, nclass=1, fchannel=-1, nf1 = 64):

    input_layer = Input(input_shape)
    input_segmentation = Input([input_shape[0], input_shape[1], 1])

    keras.initializers.glorot_uniform(seed=0)

    """ Image codification """

    l1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(input_layer)
    l1 = Activation('relu')(l1)
    l1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l1)
    l1 = Activation('relu')(l1)

    # Layer 2
    l2 = MaxPooling2D(pool_size=(2, 2))(l1)

    nf2 = 2*nf1

    l2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    l2 = Activation('relu')(l2)
    l2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    l2 = Activation('relu')(l2)

    # Layer 3
    l3 = MaxPooling2D(pool_size=(2, 2))(l2)

    nf3 = 2*nf2

    l3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    l3 = Activation('relu')(l3)
    l3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    l3 = Activation('relu')(l3)

    # Layer 4
    l4 = MaxPooling2D(pool_size=(2, 2))(l3)

    nf4 = 2*nf3

    l4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    l4 = Activation('relu')(l4)
    l4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    l4 = Activation('relu')(l4)

    """ Segmentation codification"""

    sl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(input_segmentation)
    sl1 = Activation('relu')(sl1)
    sl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(sl1)
    sl1 = Activation('relu')(sl1)

    resta_1 = Subtract(name='Capa1')([sl1,l1])

    # Layer 2
    sl2 = MaxPooling2D(pool_size=(2, 2))(sl1)

    nf2 = 2*nf1

    sl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(sl2)
    sl2 = Activation('relu')(sl2)
    sl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(sl2)
    sl2 = Activation('relu')(sl2)

    resta_2 = Subtract(name='Capa2')([sl2,l2])

    # Layer 3
    sl3 = MaxPooling2D(pool_size=(2, 2))(sl2)

    nf3 = 2*nf2

    sl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(sl3)
    sl3 = Activation('relu')(sl3)
    sl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(sl3)
    sl3 = Activation('relu')(sl3)

    resta_3 = Subtract(name='Capa3')([sl3,l3])

    # Layer 4
    sl4 = MaxPooling2D(pool_size=(2, 2))(sl3)

    nf4 = 2*nf3

    sl4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(sl4)
    sl4 = Activation('relu')(sl4)
    sl4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(sl4)
    sl4 = Activation('relu')(sl4)

    resta_4 = Subtract(name='Capa4')([sl4,l4])

    """ Image decodification """
    # Layer 3
    up3 = UpSampling2D(size=(2,2))(l4)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv3 = Conv2D(nf3, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up3)
    # End Up-conv

    # ---------- Level 3
    m_concat3 = Concatenate(axis=fchannel)([l3, upconv3])

    dl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat3)
    dl3 = Activation('relu')(dl3)
    dl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl3)
    dl3 = Activation('relu')(dl3)

    # Layer 2
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up2 = UpSampling2D(size=(2,2))(dl3)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv2 = Conv2D(nf2, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up2)
    #upconv2 = BatchNormalization(axis=fchannel)(upconv2)
    # End Up-conv

    # ---------- Level 2
    m_concat2 = Concatenate(axis=fchannel)([l2, upconv2])

    dl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat2)
    dl2 = Activation('relu')(dl2)
    dl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl2)
    dl2 = Activation('relu')(dl2)

    # Layer 1
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up1 = UpSampling2D(size=(2,2))(dl2)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv1 = Conv2D(nf1, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up1)
    # End Up-conv

    # ---------- Level 1
    m_concat1 = Concatenate(axis=fchannel)([l1, upconv1])

    dl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat1)
    dl1 = Activation('relu')(dl1)
    dl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl1)
    dl1 = Activation('relu')(dl1)

    # Final layer
    decode = Conv2D(nclass, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl1)
    decode = Activation('sigmoid', name='UNET')(decode)

    model = Model(inputs=[input_layer,input_segmentation], outputs=[decode,resta_1,resta_2,resta_3,resta_4])

    return model

"""
*************** CAPSULE COMBINATION *******************
Paper: Dynamic routing between capsules
Github: XifenGuo
"""

class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return k.sqrt(k.sum(k.square(inputs), -1) + k.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs

        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = k.sqrt(k.sum(k.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = k.one_hot(indices=k.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        """
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked
        """
        masked = k.batch_flatten(inputs)
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = k.sum(k.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / k.sqrt(s_squared_norm + k.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None): #Esto se llama cuando lo llamo asi m = y(params)(x)
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = k.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = k.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        #inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)
        def myfun(x):
            return k.batch_dot(x, self.W, [2, 3])


        inputs_hat = k.map_fn(lambda x: myfun(x), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[k.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, axis=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(k.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += k.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

"""
********** Nets using the idea of CapsNet ********
"""
def CapsNet(input_shape, numero_de_capsulas=10, routings=3, nclass = 1, nf1 = 64, fchannel = -1):
    """Paper: Dynamic Routing between Capsules"""
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=numero_de_capsulas, dim_capsule=16, routings=routings)(primarycaps)

    #y = layers.Input(shape=(numero_de_capsulas,))

    masked_by_y = Mask()(digitcaps)  # Esto lo unico que hace es un flatten

    # Shared Decoder model in training and prediction
    decoder = models.Sequential()
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*numero_de_capsulas))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape[:-1]), activation='sigmoid'))
    output_shape = (input_shape[0],input_shape[1],1)
    decoder.add(layers.Reshape(target_shape=(output_shape)))

    # Models for training and evaluation (prediction)
    model = models.Model( [x], [decoder(masked_by_y)])

    return model

def unet_and_CapsNet_Ariel(input_shape, nclass=1, fchannel=-1, nf1 = 64, numero_de_capsulas = 5, routings=3):
    # La concatenacion se produce en el unpooling del layer 2 al layer 1

    input_layer = Input(input_shape)
    keras.initializers.glorot_uniform(seed=0)

    # Layer 1
    l1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(input_layer)
    l1 = Activation('relu')(l1)
    l1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l1)

    l1 = Activation('relu')(l1)

    """ START OF THE CAPSULE """
    #from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(l1)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=numero_de_capsulas, dim_capsule=16, routings=routings)(primarycaps)

    masked_by_y = Mask()(digitcaps)  # Esto lo unico que hace es un flatten

    # Shared Decoder model in training and prediction
    decoder = models.Sequential()
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*numero_de_capsulas))
    decoder.add(layers.Dense(4092, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape[:-1]), activation='sigmoid'))
    output_shape = (input_shape[0],input_shape[1],1)
    decoder.add(layers.Reshape(target_shape=(output_shape)))

    # Models for training and evaluation (prediction)
    out_Capsule = decoder(masked_by_y)
    out_Capsule = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(out_Capsule)
    out_Capsule = Activation('relu')(out_Capsule)

    """ START OF THE U-NET"""
    # Layer 2
    l2 = MaxPooling2D(pool_size=(2, 2))(l1)

    nf2 = 2*nf1

    l2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    l2 = Activation('relu')(l2)
    l2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    l2 = Activation('relu')(l2)


    # Layer 3
    l3 = MaxPooling2D(pool_size=(2, 2))(l2)

    nf3 = 2*nf2

    l3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    l3 = Activation('relu')(l3)
    l3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    l3 = Activation('relu')(l3)

    # Layer 4
    l4 = MaxPooling2D(pool_size=(2, 2))(l3)

    nf4 = 2*nf3

    l4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    l4 = Activation('relu')(l4)
    l4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    l4 = Activation('relu')(l4)

    # Layer 3
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up3 = UpSampling2D(size=(2,2))(l4)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv3 = Conv2D(nf3, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up3)
    # End Up-conv

    # ---------- Level 3
    m_concat3 = Concatenate(axis=fchannel)([l3, upconv3])

    dl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat3)
    dl3 = Activation('relu')(dl3)
    dl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl3)
    dl3 = Activation('relu')(dl3)

    # Layer 2
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up2 = UpSampling2D(size=(2,2))(dl3)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv2 = Conv2D(nf2, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up2)
    #upconv2 = BatchNormalization(axis=fchannel)(upconv2)
    # End Up-conv

    # ---------- Level 2
    m_concat2 = Concatenate(axis=fchannel)([l2, upconv2])

    dl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat2)
    dl2 = Activation('relu')(dl2)
    dl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl2)
    dl2 = Activation('relu')(dl2)

    # Layer 1
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up1 = UpSampling2D(size=(2,2))(dl2)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv1 = Conv2D(nf1, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up1)
    # End Up-conv

    # ---------- Level 1
    m_concat1 = Concatenate(axis=fchannel)([l1, upconv1, out_Capsule])

    dl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat1)
    dl1 = Activation('relu')(dl1)
    dl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl1)

    decode = Conv2D(nclass, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl1)
    decode = Activation('sigmoid')(decode)

    model = Model(inputs=input_layer, outputs=decode)

    return model

def unet_and_CapsNet_Andres(input_shape, nclass=1, fchannel=-1, nf1 = 64, numero_de_capsulas = 5, routings=3):
    # La concatenacion de ambas redes se produce al final

    input_layer = Input(input_shape)
    keras.initializers.glorot_uniform(seed=0)

    # Layer 1

    l1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(input_layer)
    l1 = Activation('relu')(l1)
    l1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l1)
    l1 = Activation('relu')(l1)

    """ START OF THE CAPSULE """

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(l1)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=numero_de_capsulas, dim_capsule=16, routings=routings)(primarycaps)

    #y = layers.Input(shape=(numero_de_capsulas,))

    masked_by_y = Mask()(digitcaps)  # Esto lo unico que hace es un flatten

    # Shared Decoder model in training and prediction
    decoder = models.Sequential()
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*numero_de_capsulas))
    decoder.add(layers.Dense(4092, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape[:-1]), activation='sigmoid'))
    output_shape = (input_shape[0],input_shape[1],1)
    decoder.add(layers.Reshape(target_shape=(output_shape)))

    # Models for training and evaluation (prediction)
    out_Capsule = decoder(masked_by_y)

    """ START OF THE U-NET"""
    # Layer 2
    l2 = MaxPooling2D(pool_size=(2, 2))(l1)

    nf2 = 2*nf1

    l2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    l2 = Activation('relu')(l2)
    l2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l2)
    l2 = Activation('relu')(l2)

    # Layer 3
    l3 = MaxPooling2D(pool_size=(2, 2))(l2)

    nf3 = 2*nf2

    l3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    l3 = Activation('relu')(l3)
    l3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l3)
    l3 = Activation('relu')(l3)

    # Layer 4
    l4 = MaxPooling2D(pool_size=(2, 2))(l3)

    nf4 = 2*nf3

    l4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    l4 = Activation('relu')(l4)
    l4 = Conv2D(nf4, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(l4)
    l4 = Activation('relu')(l4)

    # Layer 3
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up3 = UpSampling2D(size=(2,2))(l4)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv3 = Conv2D(nf3, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up3)
    #upconv3 = BatchNormalization(axis=fchannel)(upconv3)
    # End Up-conv

    # ---------- Level 3
    m_concat3 = Concatenate(axis=fchannel)([l3, upconv3])

    dl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat3)
    dl3 = Activation('relu')(dl3)
    dl3 = Conv2D(nf3, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl3)
    dl3 = Activation('relu')(dl3)

    # Layer 2
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up2 = UpSampling2D(size=(2,2))(dl3)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv2 = Conv2D(nf2, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up2)
    # End Up-conv

    # ---------- Level 2
    m_concat2 = Concatenate(axis=fchannel)([l2, upconv2])

    dl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat2)
    dl2 = Activation('relu')(dl2)
    dl2 = Conv2D(nf2, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl2)
    dl2 = Activation('relu')(dl2)


    # Layer 1
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up1 = UpSampling2D(size=(2,2))(dl2)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv1 = Conv2D(nf1, (2,2), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(up1)
    # End Up-conv

    # ---------- Level 1
    m_concat1 = Concatenate(axis=fchannel)([l1, upconv1])

    dl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat1)
    dl1 = Activation('relu')(dl1)
    dl1 = Conv2D(nf1, (3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(dl1)
    out_Unet = Activation('relu')(dl1)

    """ CONCATENATION OF BOTH NET"""

    m_concat0 = Concatenate(axis=fchannel)([out_Unet, out_Capsule])

    decode = Conv2D(nclass, (1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(m_concat0)
    decode = Activation('sigmoid')(decode)

    model = Model(inputs=input_layer, outputs=decode)

    return model
