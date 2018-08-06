# Arda Mavi

import os
import sys
import numpy as np
from get_dataset import get_scan, scan_pading, save_seg_imgs
from keras.models import model_from_json

def predict(model, scans):

    section_size = scans.shape[-1]

    X, _ = scan_pading(scans, None, section_size = 128)

    pad_size = X.shape[-1]-section_size

    # For splitting:
    splitted_scans = []
    for i in range(0, X.shape[-1]-127, 128):
        splitted_scans.append(X[:,:,i:i+120])
    X = np.array(splitted_scans, dtype='float32')

    X = ((X-np.min(X))/(np.max(X)-np.min(X))).reshape(X.shape+(1,)) # TODO: DICOM Liver Normalization
    Y = model.predict(X)
    Y = Y.reshape(Y.shape[:-1])*255.

    images = []
    for one_img in Y:
        if images == []:
            images = one_img
        else:
            images = np.dstack((images, one_img))
    Y = images[:,:,0:images.shape[-1]-pad_size] # Remove pads from output.

    for i in range(Y.shape[-1]):
        Y[:,:,i] = np.rot90(np.fliplr(Y[:,:,i]))

    return Y

def main(dicom_path):
    if not os.path.exists(dicom_path):
        print('DICOM file not exists!')
        return None
    X = get_scan(dicom_path, scan_size = (128, 128))

    # Getting model:
    with open('Data/GAN-Models/Generator/model.json', 'r') as model_file:
        model = model_file.read()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/GAN-Models/Generator/weights.h5")

    Y = predict(model, X)
    return Y

if __name__ == '__main__':
    dicom_path = sys.argv[1]
    Y = main(dicom_path)
    if type(Y) != type(None):
        save_path = 'Data/Segmented'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_seg_imgs(Y, save_path)
    else:
        print('None Output Error!')
