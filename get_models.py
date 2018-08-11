# Arda Mavi

import os
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adadelta
from keras.utils import multi_gpu_model
from keras.models import model_from_json
from keras.layers import Input, Conv3D, Dense, UpSampling3D, Activation, MaxPooling3D, Dropout, concatenate, Flatten, Multiply

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
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Segment Model:
def get_segment_model(data_shape):
    # U-Net:
    inputs = Input(shape=(data_shape))

    conv_block_1 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
    conv_block_1 = Activation('relu')(conv_block_1)
    conv_block_1 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_1)
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

    encoder = Model(inputs=inputs, outputs=conv_block_5)

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
    up_block_4 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(up_block_4)

    merge_4 = concatenate([conv_block_1, up_block_4])

    conv_block_9 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(merge_4)
    conv_block_9 = Activation('relu')(conv_block_9)
    conv_block_9 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(conv_block_9)
    conv_block_9 = Activation('relu')(conv_block_9)

    conv_block_10 = Conv3D(data_shape[-1], (1, 1, 1), strides=(1, 1, 1), padding='same')(conv_block_9)
    outputs = Activation('sigmoid')(conv_block_10)

    model = Model(inputs=inputs, outputs=outputs)

    """
    # For Multi-GPU:

    try:
        model = multi_gpu_model(model)
    except:
        pass
    """

    model.compile(optimizer = Adadelta(lr=0.001), loss='mse', metrics=['acc'])

    print('Segment Model Architecture:')
    print(model.summary())

    return model, encoder

# GAN:
def get_GAN(input_shape, Generator, Discriminator):
    input_gan = Input(shape=(input_shape))
    generated_seg = Generator(input_gan)
    gan_output = Discriminator([input_gan, generated_seg])

    # Compile GAN:
    gan = Model(input_gan, gan_output)
    gan.compile(optimizer=Adadelta(lr=0.001), loss='mse', metrics=['accuracy'])

    print('GAN Architecture:')
    print(gan.summary())
    return gan

def get_Generator(input_shape):
    Generator, _ = get_segment_model(input_shape)
    print('Generator Architecture:')
    print(Generator.summary())
    return Generator

def get_Discriminator(input_shape_1, input_shape_2, Encoder):

    dis_inputs_1 = Input(shape=input_shape_1) # For Encoder Model
    dis_inputs_2 = Input(shape=input_shape_2) # From Segmentated Image

    encoder_output_1 = Encoder(dis_inputs_1)

    mul_1 = Multiply()([dis_inputs_1, dis_inputs_2])

    encoder_output_2 = Encoder(mul_1)

    merge_dis = concatenate([encoder_output_1, encoder_output_2])

    dis_conv_block = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same')(merge_dis)
    dis_conv_block = Activation('relu')(dis_conv_block)
    dis_conv_block = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same')(dis_conv_block)
    dis_conv_block = Activation('relu')(dis_conv_block)
    dis_conv_block = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(dis_conv_block)

    dis_conv_block = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same')(dis_conv_block)
    dis_conv_block = Activation('relu')(dis_conv_block)
    dis_conv_block = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same')(dis_conv_block)
    dis_conv_block = Activation('relu')(dis_conv_block)

    dis_conv_block = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(dis_conv_block)
    dis_conv_block = Activation('relu')(dis_conv_block)
    dis_conv_block = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(dis_conv_block)
    dis_conv_block = Activation('relu')(dis_conv_block)

    flat_1 = Flatten()(dis_conv_block)

    dis_fc_1 = Dense(512)(flat_1)
    dis_fc_1 = Activation('relu')(dis_fc_1)
    dis_fc_1 = Dense(512)(dis_fc_1)
    dis_fc_1 = Activation('relu')(dis_fc_1)

    dis_drp_1 = Dropout(0.2)(dis_fc_1)

    dis_fc_2 = Dense(256)(dis_drp_1)
    dis_fc_2 = Activation('relu')(dis_fc_2)

    dis_drp_2 = Dropout(0.2)(dis_fc_2)

    dis_fc_3 = Dense(64)(dis_drp_2)
    dis_fc_3 = Activation('relu')(dis_fc_3)

    dis_drp_3 = Dropout(0.2)(dis_fc_3)

    dis_fc_4 = Dense(1)(dis_drp_3)
    dis_similarity_output = Activation('sigmoid')(dis_fc_4)

    Discriminator = Model(inputs=[dis_inputs_1, dis_inputs_2], outputs=dis_similarity_output)
    Discriminator.compile(optimizer=Adadelta(lr=0.001), loss='mse', metrics=['accuracy'])

    print('Discriminator Architecture:')
    print(Discriminator.summary())
    return Discriminator

if __name__ == '__main__':
    segment_model, encoder = get_segment_model((128,128,128,1))
    generator = get_Generator((128,128,128,1))
    discriminator = get_Discriminator((128,128,128,1), (128,128,128,1), encoder)
    gan = get_GAN((128,128,128,1), generator, discriminator)
