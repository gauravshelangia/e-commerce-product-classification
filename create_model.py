from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def get_base_model():
    """
    Returns the convolutional part of VGG net as a keras model
    All layers have trainable set to False
    """
    """"
    img_width and img_height can vary according to one's need
    """
    img_width, img_height = 224, 224

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height,3), name='image_input'))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2),dim_ordering="th"))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2),dim_ordering="th"))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2),dim_ordering="th"))
    #
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2),dim_ordering="th"))
    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))

    # # set trainable to false in all layers
    # for layer in model.layers:
    #     if hasattr(layer, 'trainable'):
    #         layer.trainable = False
    #
    # return model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def load_weights_in_base_model(model):
    """
    The function takes the VGG convolutian part and loads
    the weights from the pre-trained model and then returns the model
    """
    weight_file = ''.join((WEIGHTS_PATH, 'vgg16_weights.h5'))
    f = h5py.File(weight_file)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    return model
