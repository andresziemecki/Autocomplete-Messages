#  import os
import numpy as np
import tensorflow as tf

# Set Seeds for repetibility
np.random.seed(1)
tf.random.set_seed(2)

import tensorflow.keras as keras
import tensorflow.keras.backend as k
from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D, Activation, Lambda
#  from tensorflow.keras.layers import UpSampling2D, Dense, Dropout, Activation, Flatten
#  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Add, Input, Subtract
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate,Input, Subtract
#  from tensorflow.keras.layers.normalization import BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import multi_gpu_model
# from tensorflow.keras.models import load_model
#  from tensorflow.keras import models, optimizers, initializers, layers


#  from tensorflow.python.ops.parallel_for import batch_jacobian as tf_batch_jacobian

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return k.sqrt(k.sum(k.square(y_pred - y_true), axis=-1))

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
        possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + k.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
        predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + k.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+k.epsilon()))

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
    intersection = k.sum(tf.math.multiply(y_true_f, y_pred_f))
    denom = k.sum(y_true_f) + k.sum(y_pred_f) - intersection
    j =  (intersection + smooth) / (denom + smooth)

    return 1 - j


# U-Net Architecture with 4 levels of encode-decode
def unet(input_shape, nclass=1, fchannel=-1, nf1 = 64, suffix=''):

    input_layer = Input(input_shape,name = 'Input_Image'+suffix)
    keras.initializers.glorot_uniform(seed=0)
    from tensorflow.keras.regularizers import l2

    # Layer 1
    # NOTE: NO esta bueno que tengan la misma semilla, esto debería hacer que
    # la inicialización sea casi igual de todos los kernels, al menos igual
    # para los que tienen la misma cantidad de elementos.
    l1 = Conv2D(nf1, (3, 3),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01),padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0),
            name = 'l1_Conv2D_1'+suffix)(input_layer)
    l1 = Activation('relu', name = 'l1_Activation_1'+suffix)(l1)
    l1 = Conv2D(nf1, (3, 3),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'l1_Conv2D_2'+suffix)(l1)
    l1 = Activation('relu', name = 'l1_Activation_2'+suffix)(l1)

    # Layer 2
    L2 = MaxPooling2D(pool_size=(2, 2), name = 'l2_MaxPooling2D'+suffix)(l1)

    nf2 = 2*nf1
    L2 = Conv2D(nf2, (3, 3),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'l2_Conv2D_1'+suffix)(L2)
    L2 = Activation('relu', name = 'l2_Activation_1'+suffix)(L2)
    L2 = Conv2D(nf2, (3, 3),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'l2_Conv2D_2'+suffix)(L2)
    L2 = Activation('relu', name = 'l2_Activation_2'+suffix)(L2)

    # Layer 3
    l3 = MaxPooling2D(pool_size=(2, 2), name = 'l3_MaxPooling2D'+suffix)(L2)

    nf3 = 2*nf2
    l3 = Conv2D(nf3, (3, 3),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'l3_Conv2D_1'+suffix)(l3)
    l3 = Activation('relu', name = 'l3_Activation_1'+suffix)(l3)
    l3 = Conv2D(nf3, (3, 3),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'l3_Conv2D_2'+suffix)(l3)
    l3 = Activation('relu', name = 'l3_Activation_2'+suffix)(l3)

    # Layer 4

    l4 = MaxPooling2D(pool_size=(2, 2), name = 'l4_MaxPooling2D'+suffix)(l3)

    nf4 = 2*nf3
    l4 = Conv2D(nf4, (3, 3),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'l4_Conv2D_1'+suffix)(l4)
    l4 = Activation('relu', name = 'l4_Activation_1'+suffix)(l4)
    l4 = Conv2D(nf4, (3, 3),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'l4_Conv2D_2'+suffix)(l4)
    l4 = Activation('relu', name = 'l4_Activation_2'+suffix)(l4)

    # Layer 3
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up3 = UpSampling2D(size=(2,2), name = 'up3_UpSampling2D'+suffix)(l4)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv3 = Conv2D(nf3, (2,2),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'upconv3_Conv2D_1'+suffix)(up3)
    # End Up-conv

    # ---------- Level 3
    m_concat3 = Concatenate(axis=fchannel, name = 'm_concat3'+suffix)([l3, upconv3])

    dl3 = Conv2D(nf3, (3, 3),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'dl3_Conv2D_1'+suffix)(m_concat3)
    dl3 = Activation('relu', name = 'dl3_Activation_1'+suffix)(dl3)
    dl3 = Conv2D(nf3, (3, 3),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'dl3_Conv2D_2'+suffix)(dl3)
    dl3 = Activation('relu', name = 'dl3_Activation_2'+suffix)(dl3)

    # Layer 2
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up2 = UpSampling2D(size=(2,2), name = 'up2_UpSampling2D'+suffix)(dl3)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv2 = Conv2D(nf2, (2,2),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'upconv2_Conv2D'+suffix)(up2)
    #upconv2 = BatchNormalization(axis=fchannel)(upconv2)
    # End Up-conv

    # ---------- Level 2
    m_concat2 = Concatenate(axis=fchannel, name = 'm_concat2'+suffix)([L2, upconv2])

    dl2 = Conv2D(nf2, (3, 3),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'dl2_Conv2D_1'+suffix)(m_concat2)
    dl2 = Activation('relu', name = 'dl2_Activation_1'+suffix)(dl2)
    dl2 = Conv2D(nf2, (3, 3),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'dl2_Conv2D_2'+suffix)(dl2)
    dl2 = Activation('relu', name = 'dl2_Activation_2'+suffix)(dl2)

    # Layer 1
    # ------ Move to next level
    # Up-conv of sampling (2x2)
    up1 = UpSampling2D(size=(2,2), name = 'up1_UpSampling2D'+suffix)(dl2)
    # Convolution without activation, so, a(x) = x which is equivalent to
    # a linear activation
    upconv1 = Conv2D(nf1, (2,2),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'upconv1_Conv2D'+suffix)(up1)
    #upconv1 = BatchNormalization(axis=fchannel)(upconv1)
    # End Up-conv

    # ---------- Level 1
    m_concat1 = Concatenate(axis=fchannel, name='m_concat1'+suffix)([l1, upconv1])

    dl1 = Conv2D(nf1, (3, 3),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'dl1_Conv2D_1'+suffix)(m_concat1)
    dl1 = Activation('relu', name = 'dl1_Activation_1'+suffix)(dl1)
    dl1 = Conv2D(nf1, (3, 3),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'dl1_Conv2D_2'+suffix)(dl1)
    dl1 = Activation('relu', name = 'dl1_Activation_2'+suffix)(dl1)

    # Final layer
    decode = Conv2D(nclass, (1, 1),  kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), padding='same', 
            kernel_initializer=keras.initializers.glorot_uniform(seed=0), 
            name = 'decode_Conv2D'+suffix)(dl1)
    decode = Activation('sigmoid', name = 'decode_Activation'+suffix)(decode)

    model = Model(inputs=input_layer, outputs=decode)

    return model


def ACNN(input_shape, encoder_model, nclass=1, fchannel=-1, nf1 = 64):

    #Load the U-Net, then connect the encoder part at the end of the U-Net
    model = unet(input_shape=input_shape, nclass=nclass, fchannel = fchannel, nf1 = nf1)

    # We have to change all names from all layers in the encoder_model because it has the same names as the model
    for i in range(len(encoder_model.layers)):
        # Podrías usar name + = '_encoder_...'
        encoder_model.layers[i].name =encoder_model.layers[i].name + '_encoder_segmentation'

    # Insert the output of the encoder to the unet
    em = encoder_model(model.outputs)
    model = Model(model.inputs, em)

    """ OutPut Encode """

    #Clone the encoder model for the input of the image segmentation
    encoderModelOutput = keras.models.clone_model(encoder_model)

    # changing the names ti th
    for i in range(len(encoderModelOutput.layers)):
        encoderModelOutput.layers[i].name =encoderModelOutput.layers[i].name + '_encoder_output'

    resta = Subtract(name = 'salida')([model.outputs[-1], encoderModelOutput.outputs[-1]])

    finalModel = Model(inputs=[model.inputs[0], encoderModelOutput.inputs[0]],outputs=[model.get_layer("decode_Activation").output, resta])

    return finalModel


def ACNN_ariel(input_shape, encoder_model, nclass=1, fchannel=-1, nf1=64):
    #Load the U-Net, then connect the encoder part at the end of the U-Net
    mUnet = unet(input_shape=input_shape, nclass=nclass, fchannel = fchannel, 
            nf1 = nf1, suffix='_unet')
    # Creamos el modelo uniendo la salida de la unet con la entrada del encoder
    lout = encoder_model(mUnet.outputs)
    model = keras.models.Model(inputs=mUnet.inputs, 
            outputs=[mUnet.output,lout])

    enc2 =  keras.models.clone_model(encoder_model)
    lout2 = Lambda(lambda x: keras.backend.abs(x), name='salida')(lout - enc2.output)

    mACCN = keras.models.Model(inputs=[model.input, enc2.input], 
            outputs=[model.outputs[0], lout2])

    return mACCN
