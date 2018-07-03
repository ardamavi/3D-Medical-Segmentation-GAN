# Arda Mavi

import os
from keras.models import Model
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.models import model_from_json
from keras.layers import Input, Conv3D, UpSampling3D, Activation, MaxPooling3D, concatenate

def save_model(model, path='Data/Model/', model_name = 'model', weights_name = 'weights'):
    if not os.path.exists(path):
        os.makedirs(path)
    model_json = model.to_json()
    with open(path+model_name+'.json', 'w') as model_file:
        model_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path+weights_name+'.h5')
    print('Model and weights saved to ' + path+model_name+'.json and ' + path+weights_name+'.h5')
    return

def get_model(model_path, weights_path):
    if not os.path.exists(model_path):
        print('Model file not exists!')
        return None
    elif not os.path.exists(weights_path):
        print('Weights file not exists!')
        return None

    # Getting model:
    with open(model_path, 'r') as model_file:
        model = model_file.read()
    model = model_from_json(model)
    # Getting weights
    model.load_weights(weights_path)
    return model

# Loss Function:
def dice_coefficient(y_true, y_pred):
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return -2. * K.sum(flat_y_true * flat_y_pred) / (K.sum(flat_y_true) + K.sum(flat_y_pred))

def dice_coefficient_loss(y_true, y_pred):
    return - dice_coefficient(y_true, y_pred)

def get_unet(data_shape):

    inputs = Input(shape=(data_shape))

    conv_block_1 = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
    conv_block_1 = Activation('relu')(conv_block_1)
    conv_block_1 = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_1)
    conv_block_1 = Activation('relu')(conv_block_1)
    pool_block_1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_block_1)

    conv_block_2 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(pool_block_1)
    conv_block_2 = Activation('relu')(conv_block_2)
    conv_block_2 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_2)
    conv_block_2 = Activation('relu')(conv_block_2)
    pool_block_2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_block_2)

    conv_block_3 = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same')(pool_block_2)
    conv_block_3 = Activation('relu')(conv_block_3)
    conv_block_3 = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_3)
    conv_block_3 = Activation('relu')(conv_block_3)
    pool_block_3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_block_3)

    conv_block_4 = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same')(pool_block_3)
    conv_block_4 = Activation('relu')(conv_block_4)
    conv_block_4 = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_4)
    conv_block_4 = Activation('relu')(conv_block_4)
    pool_block_4 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_block_4)

    conv_block_5 = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same')(pool_block_4)
    conv_block_5 = Activation('relu')(conv_block_5)
    conv_block_5 = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_5)
    conv_block_5 = Activation('relu')(conv_block_5)

    up_block_1 = UpSampling3D((2, 2, 2))(conv_block_5)
    up_block_1 = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same')(up_block_1)

    merge_1 = concatenate([conv_block_4, up_block_1])

    conv_block_6 = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same')(merge_1)
    conv_block_6 = Activation('relu')(conv_block_6)
    conv_block_6 = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_6)
    conv_block_6 = Activation('relu')(conv_block_6)

    up_block_2 = UpSampling3D((2, 2, 2))(conv_block_6)
    up_block_2 = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same')(up_block_2)

    merge_2 = concatenate([conv_block_3, up_block_2])

    conv_block_7 = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same')(merge_2)
    conv_block_7 = Activation('relu')(conv_block_7)
    conv_block_7 = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_7)
    conv_block_7 = Activation('relu')(conv_block_7)

    up_block_3 = UpSampling3D((2, 2, 2))(conv_block_7)
    up_block_3 = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same')(up_block_3)

    merge_3 = concatenate([conv_block_2, up_block_3])

    conv_block_8 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(merge_3)
    conv_block_8 = Activation('relu')(conv_block_8)
    conv_block_8 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_8)
    conv_block_8 = Activation('relu')(conv_block_8)

    up_block_4 = UpSampling3D((2, 2, 2))(conv_block_8)
    up_block_4 = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same')(up_block_4)

    merge_4 = concatenate([conv_block_1, up_block_4])

    conv_block_9 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(merge_4)
    conv_block_9 = Activation('relu')(conv_block_9)
    conv_block_9 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_9)
    conv_block_9 = Activation('relu')(conv_block_9)

    conv_block_10 = Conv3D(data_shape[-1], (1, 1, 1), strides=(1, 1, 1), padding='same')(conv_block_9)
    outputs = Activation('sigmoid')(conv_block_10)

    model = Model(inputs=inputs, outputs=outputs)

    try:
        model = multi_gpu_model(model)
    except:
        pass

    model.compile(optimizer = 'adadelta', loss=dice_coefficient_loss, metrics=[dice_coefficient])

    return model

# TODO: GAN

if __name__ == '__main__':
    model = get_unet((512,512,16,1))
    print(model.summary())
    save_model(model, path='Data/Model/', model_name = 'model', weights_name = 'weights')
    print('Model saved to "Data/Model"!')
