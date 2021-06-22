from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, Dropout, concatenate
from keras.models import Model
from keras.optimizers import *

# function that defines one convolutional layer with certain number of filters
def _single_conv(input_tensor, n_filters, kernel_size):
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), activation = 'sigmoid')(input_tensor)
    return x

# function that defines two sequential 2D convolutional layers with certain number of filters
def _double_conv(input_tensor, n_filters, kernel_size = 3):
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same', kernel_initializer = 'he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# function that defines 2D transposed convolutional (Deconvolutional) layer
def _deconv(input_tensor, n_filters, kernel_size = 3, stride = 2):
    x = Conv2DTranspose(filters = n_filters, kernel_size = (kernel_size, kernel_size), strides = (stride, stride), padding = 'same')(input_tensor)
    return x

# function that defines Max Pooling layer with pool size 2 and applies Dropout
def _pooling(input_tensor, dropout_rate = 0.1):
    x = MaxPooling2D(pool_size = (2, 2))(input_tensor)
    x = Dropout(rate = dropout_rate)(x)
    return x

# function that merges two layers (Concatenate)
def _merge(input1, input2):
    x = concatenate([input1, input2])
    return x

def UNet(input_size, n_filters):
    ''' U-Net atchitecture
    Creating a U-Net class that inherits from keras.models.Model
    In initializer, CNN layers are defined using functions from model.utils
    Then parent-initializer is called wuth calculated input and output layers
    Build function is also defined for model compilation and summary
    checkpoint returns a ModelCheckpoint for best model fitting
    '''
    
    # input layer
    input = Input(input_size)

    # contraction path
    conv1 = _double_conv(input, n_filters * 1)
    pool1 = _pooling(conv1)

    conv2 = _double_conv(pool1, n_filters * 2)
    pool2 = _pooling(conv2)

    conv3 = _double_conv(pool2, n_filters * 4)
    pool3 = _pooling(conv3)

    conv4 = _double_conv(pool3, n_filters * 8)
    pool4 = _pooling(conv4)

    conv5 = _double_conv(pool4, n_filters * 16)

    # expansive path
    up6 = _deconv(conv5, n_filters * 8)
    up6 = _merge(conv4, up6)
    conv6 = _double_conv(up6, n_filters * 8)

    up7 = _deconv(conv6, n_filters * 4)
    up7 = _merge(conv3, up7)
    conv7 = _double_conv(up7, n_filters * 4)

    up8 = _deconv(conv7, n_filters * 2)
    up8 = _merge(conv2, up8)
    conv8 = _double_conv(up8, n_filters * 2)

    up9 = _deconv(conv8, n_filters * 1)
    up9 = _merge(conv1, up9)
    conv9 = _double_conv(up9, n_filters * 1)

    # output layer
    output = _single_conv(conv9, 1, 1)

    # model
    model = Model(inputs = input, outputs = output)

    # compile
    model.compile(optimizer=Adam(),
                  loss="binary_crossentropy", 
                  metrics=["accuracy"])
        
    return model