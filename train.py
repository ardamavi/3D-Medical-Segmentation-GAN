# Arda Mavi

import os
import numpy as np
from os import listdir
from random import shuffle
from get_dataset import read_npy_dataset, split_npy_dataset
from get_models import get_segment_model, save_model
# from keras.callbacks import ModelCheckpoint, TensorBoard

epochs = 25
# batch_size = 1
test_size = 0.2

# Training Segment Model:
def train_seg_model(model, splitted_npy_dataset_path='Data/npy_dataset/splitted_npy_dataset', test_path = 'Data/npy_dataset/test_npy', epochs):
    if not os.path.exists('Data/Checkpoints/'):
        os.makedirs('Data/Checkpoints/')
    # checkpoints = []
    # checkpoints.append(TensorBoard(log_dir='Data/Checkpoints/./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

    test_XY = np.load(test_path+'/test.npy')
    X_test, Y_test = test_XY[0], test_XY[1]
    batch_dirs = listdir(splitted_npy_dataset_path)
    for epoch in range(epochs):
        print('Epoch: {0}/{1}'.format(epoch+1, epochs))
        shuffle(batch_dirs)
        for batch_path in batch_dirs:
            batch_XY = np.load(splitted_npy_dataset_path+'/'+batch_path)
            X_batch, Y_batch = batch_XY[0], batch_XY[1]
            # model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=1, callbacks=checkpoints)
            model.train_on_batch(X_batch, Y_batch)
        scores = model.evaluate(X_test, Y_test) # TODO: Optimize memory ueses!
        print('Test loss:', scores[0], 'Test accuracy:', scores[1])
        model.save_weights('Data/Checkpoints/last_weights.h5')
    return model

# Training GAN:
def train_gan(Generator, Encoder, Discriminator, GAN, splitted_npy_dataset_path='Data/npy_dataset/splitted_npy_dataset', test_path = 'Data/npy_dataset/test_npy', epochs):
    # TODO
    return Generator, Encoder, Discriminator

def main(train_gan_model = 1):
    if train_gan_model:
        # Getting Generator:
        Generator, Encoder = get_segment_model(data_shape = (256, 256, 32, 1))
        Discriminator = get_Discriminator(input_shape_1 = (16,16,2,256), input_shape_2 = (256, 256, 32, 1))
        GAN = get_GAN((16,16,2,256), (256, 256, 32, 1), Generator, Discriminator)

        # Saving non-trained models:
        save_model(Generator, path='Data/GAN-Models/Generator', model_name = 'model', weights_name = 'weights')
        save_model(Encoder, path='Data/GAN-Models/Encoder', model_name = 'model', weights_name = 'weights')
        save_model(Discriminator, path='Data/GAN-Models/Discriminator', model_name = 'model', weights_name = 'weights')
        print('Non-Trained model saved to "Data/GAN-Models"!')

        # Train:
        Generator, Encoder, Discriminator = train_gan(Generator, Encoder, Discriminator, GAN, splitted_npy_dataset_path='Data/npy_dataset/splitted_npy_dataset', test_path = 'Data/npy_dataset/test_npy', epochs = 100)

        # Saving trained models:
        save_model(Generator, path='Data/GAN-Models/Generator', model_name = 'model', weights_name = 'weights')
        save_model(Encoder, path='Data/GAN-Models/Encoder', model_name = 'model', weights_name = 'weights')
        save_model(Discriminator, path='Data/GAN-Models/Discriminator', model_name = 'model', weights_name = 'weights')
        print('Trained model saved to "Data/GAN-Models"!')
        return Generator
    else:
        segment_model, _ = get_segment_model(data_shape = (256, 256, 32, 1))
        print(segment_model.summary())
        save_model(segment_model, path='Data/Model/', model_name = 'model', weights_name = 'weights')
        print('Non-Trained model saved to "Data/Model"!')
        model = train_seg_model(segment_model, splitted_npy_dataset_path='Data/npy_dataset/splitted_npy_dataset', test_path = 'Data/npy_dataset/test_npy', epochs = 25)
        save_model(segment_model, path='Data/Model/', model_name = 'model', weights_name = 'weights')
        print('Trained model saved to "Data/Model"!')
        return segment_model

if __name__ == '__main__':
    main(train_gan_model = 0)
