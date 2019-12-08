from keras.models import Model
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.initializers import glorot_uniform
from keras.regularizers import l2

# Half-U-Net Architecture with 4 levels of encode-decode
def digit_recognizer_model(input_shape=(28,28,1), kernel_size=(5,5), nf1 = 32, suffix=''):

    input_layer = Input(input_shape)
    seed = 0

    # Layer 1 (28x28)
    l1 = Conv2D(filters = nf1,
        kernel_size = kernel_size,
        kernel_regularizer=l2(0.01), 
        bias_regularizer=l2(0.01),
        padding='valid', 
        kernel_initializer=glorot_uniform(seed=seed),
        name = 'l1_Conv2D_1')(input_layer)
    seed += 1

    l1 = Activation('relu', name = 'l1_Activation_1'+suffix)(l1)
    
    # (26x26) 
    l1 =Conv2D(filters = nf1,
        kernel_size = kernel_size,
        kernel_regularizer=l2(0.01), 
        bias_regularizer=l2(0.01),
        padding='valid', 
        kernel_initializer=glorot_uniform(seed=seed),
        name = 'l1_Conv2D_2')(l1)
    seed += 1

    l1 = Activation('relu', name = 'l1_Activation_2')(l1)

    # Layer 2 (24x24)
    L2 = MaxPooling2D(pool_size=(2, 2),
        name = 'l2_MaxPooling2D')(l1)

    L2 = Dropout(rate = 0.25)(L2)

    # (12x12)
    nf2 = 2*nf1
    L2 = Conv2D(filters = nf2,
        kernel_size = (3,3),
        kernel_regularizer=l2(0.01), 
        bias_regularizer=l2(0.01),
        padding='valid', 
        kernel_initializer=glorot_uniform(seed=seed), 
        name = 'l2_Conv2D_1'+suffix)(L2)
    seed += 1

    
    L2 = Activation('relu', name = 'l2_Activation_1'+suffix)(L2)

    # (10x10)
    L2 = Conv2D(filters = nf2,
        kernel_size = (3,3),
        kernel_regularizer=l2(0.01), 
        bias_regularizer=l2(0.01),
        padding='valid', 
        kernel_initializer=glorot_uniform(seed=seed), 
        name = 'l2_Conv2D_2'+suffix)(L2)
    seed += 1

    L2 = Activation('relu', name = 'l2_Activation_2'+suffix)(L2)

    # Layer 3 (8x8)
    l3 = MaxPooling2D(pool_size=(2, 2),
        name = 'l3_MaxPooling2D'+suffix)(L2)

    l3 = Dropout(rate=0.25)(l3)

    l4 = Flatten()(l3)

    l5 = Dense(256, activation='relu')(l4)

    l5 = Dropout(rate=0.5)(l5)

    l5 = Dense(10, activation='softmax')(l4)

    model = Model(inputs=input_layer, outputs=l5)

    return model
