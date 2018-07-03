# Arda Mavi

import os
from get_dataset import read_npy_dataset
from get_models import get_unet, save_model
from keras.callbacks import ModelCheckpoint, TensorBoard

epochs = 25
batch_size = 8

# TODO: Train GAN

def train_model(model, X, X_test, Y, Y_test):
    if not os.path.exists('Data/Checkpoints/'):
        os.makedirs('Data/Checkpoints/')
    checkpoints = []
    checkpoints.append(ModelCheckpoint('Data/Checkpoints/best_weights.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
    checkpoints.append(TensorBoard(log_dir='Data/Checkpoints/./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

    model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test), shuffle=True, callbacks=checkpoints)

    return model

def main():
    X, X_test, Y, Y_test = read_npy_dataset('Data/npy_dataset', test_size=0.2)
    model = get_unet(data_shape = (512, 512, 16, 1))
    print(model.summary())
    save_model(model, path='Data/Model/', model_name = 'model', weights_name = 'weights')
    print('Non-Trained model saved to "Data/Model"!')
    model = train_model(model, X, X_test, Y, Y_test)
    save_model(model, path='Data/Model/', model_name = 'model', weights_name = 'weights')
    print('Trained model saved to "Data/Model"!')
    return model

if __name__ == '__main__':
    main()
